//! Smart builders for union and intersection types.
//!
//! Invariants we maintain here:
//!   * No single-element union types (should just be the contained type instead.)
//!   * No single-positive-element intersection types. Single-negative-element are OK, we don't
//!     have a standalone negation type so there's no other representation for this.
//!   * The same type should never appear more than once in a union or intersection. (This should
//!     be expanded to cover subtyping -- see below -- but for now we only implement it for type
//!     identity.)
//!   * Disjunctive normal form (DNF): the tree of unions and intersections can never be deeper
//!     than a union-of-intersections. Unions cannot contain other unions (the inner union just
//!     flattens into the outer one), intersections cannot contain other intersections (also
//!     flattens), and intersections cannot contain unions (the intersection distributes over the
//!     union, inverting it into a union-of-intersections).
//!   * No type in a union can be a subtype of any other type in the union (just eliminate the
//!     subtype from the union).
//!   * No type in an intersection can be a supertype of any other type in the intersection (just
//!     eliminate the supertype from the intersection).
//!   * An intersection containing two non-overlapping types simplifies to [`Type::Never`].
//!
//! The implication of these invariants is that a [`UnionBuilder`] does not necessarily build a
//! [`Type::Union`]. For example, if only one type is added to the [`UnionBuilder`], `build()` will
//! just return that type directly. The same is true for [`IntersectionBuilder`]; for example, if a
//! union type is added to the intersection, it will distribute and [`IntersectionBuilder::build`]
//! may end up returning a [`Type::Union`] of intersections.
//!
//! ## Performance
//!
//! In practice, there are two kinds of unions found in the wild: relatively-small unions made up
//! of normal user types (classes, etc), and large unions made up of literals, which can occur via
//! large enums or from string/integer/bytes literals, which can grow due to literal arithmetic or
//! operations on literal strings/bytes. For normal unions, it's most efficient to just store the
//! member types in a vector, and do O(n^2) redundancy checks to maintain the union in simplified
//! form. But literal unions can grow to a size where this becomes a performance problem. For this
//! reason, we group literal types in `UnionBuilder`. Since every different string literal type
//! shares exactly the same possible super-types, and none of them are subtypes of each other
//! (unless exactly the same literal type), we can avoid many unnecessary redundancy checks.

use super::RecursivelyDefined;
use crate::types::enums::{EnumComplement, enum_metadata};
use crate::types::set_theoretic::expand_intersection_typevars_and_newtypes;
use crate::types::tuple::{TupleSpec, TupleSpecBuilder, TupleType};
use crate::types::visitor::any_over_type;
use crate::types::{
    BytesLiteralType, ClassLiteral, EnumLiteralType, IntersectionType, KnownClass,
    LiteralValueType, LiteralValueTypeKind, NegativeIntersectionElements, StringLiteralType,
    SubclassOfType, Type, TypeVarBoundOrConstraints, UnionType,
};
use crate::{Db, FxOrderMap, FxOrderSet};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

/// Extract `(core, guard)` from truthiness-guarded intersections.
///
/// e.g.
/// - `A & ~AlwaysTruthy` -> `Some((A, ~AlwaysTruthy))`
/// - `A & ~AlwaysFalsy` -> `Some((A, ~AlwaysFalsy))`
/// - `A` -> `None`
/// - `A & ~AlwaysTruthy & ~AlwaysFalsy` -> `None` (not a single-guard shape)
///
/// This only recognizes the "single truthiness guard" forms used by truthiness narrowing.
fn split_truthiness_guarded_intersection<'db>(
    db: &'db dyn Db,
    ty: Type<'db>,
) -> Option<(Type<'db>, Type<'db>)> {
    let Type::Intersection(intersection) = ty else {
        return None;
    };
    let falsy = Type::AlwaysTruthy.negate(db);
    let truthy = Type::AlwaysFalsy.negate(db);

    let has_not_truthy = intersection.negative(db).contains(&Type::AlwaysTruthy);
    let has_not_falsy = intersection.negative(db).contains(&Type::AlwaysFalsy);
    let guard = match (has_not_truthy, has_not_falsy) {
        (true, false) => falsy,
        (false, true) => truthy,
        _ => return None,
    };

    let mut core = IntersectionBuilder::new(db);
    for positive in intersection.positive(db) {
        core = core.add_positive(*positive);
    }
    for negative in intersection.negative(db) {
        if (guard == falsy && *negative == Type::AlwaysTruthy)
            || (guard == truthy && *negative == Type::AlwaysFalsy)
        {
            continue;
        }
        core = core.add_negative(*negative);
    }
    Some((core.build(), guard))
}

/// Return whether tuple or protocol elements can be refined set-theoretically.
///
/// Dynamic types can be hidden behind aliases, so checking only the top-level types is insufficient.
fn all_elements_are_static(db: &dyn Db, elements: &[Type<'_>]) -> bool {
    elements
        .iter()
        .all(|element| !any_over_type(db, *element, true, |ty| ty.is_dynamic()))
}

/// Build an intersection used while planning indexed-protocol complements.
///
/// Planning can simplify ordinary intersections, but must leave nested indexed-protocol
/// complements symbolic so that tuple materialization remains owned by the outer plan.
fn build_indexed_protocol_planning_intersection<'db>(
    db: &'db dyn Db,
    positives: impl IntoIterator<Item = Type<'db>>,
    negatives: impl IntoIterator<Item = Type<'db>>,
) -> Type<'db> {
    let mut builder = IntersectionBuilder::new(db).positive_elements(positives);
    for negative in negatives {
        builder = builder.add_negative(negative);
    }
    builder.build_without_indexed_protocol_distribution()
}

/// Limit the total number of intersection alternatives produced by tuple protocol complements.
const MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES: usize = 128;

#[derive(Debug, Eq, PartialEq)]
enum TupleProtocolComplementPlan<'db> {
    NotApplicable,
    ExceedsLimit,
    Unchanged,
    Eliminated,
    Alternatives(Vec<(usize, Type<'db>)>),
}

#[derive(Debug, Clone)]
enum InnerIndexedProtocolComplementPlan<'db> {
    Unchanged,
    Eliminated,
    Alternatives {
        tuple_changes: Vec<Vec<(usize, usize, Type<'db>)>>,
    },
}

#[derive(Debug)]
struct TupleReplacementPlan<'db> {
    positive_index: usize,
    elements: Vec<Type<'db>>,
}

#[derive(Debug)]
struct InnerIndexedProtocolComplementsPlan<'db> {
    handled_protocols: FxOrderSet<Type<'db>>,
    alternatives: Vec<Vec<TupleReplacementPlan<'db>>>,
}

/// Refine a tuple specialization using finite indexed constraints from a protocol.
///
/// Returns `None` when `ty` is not a tuple specialization, `Some(Never)` when the tuple shape is
/// disjoint from the protocol, and the refined tuple type otherwise.
///
/// This intentionally assumes that tuple subclasses preserve the relationship between iteration
/// and indexing provided by the builtin `tuple` class. A tuple subclass can override `__iter__` so
/// that a sequence pattern observes different element types than inherited indexing does, making
/// this refinement unsound for that subclass. We accept this limitation to retain precise
/// narrowing for ordinary tuple annotations.
fn refine_tuple_with_indexed_protocol<'db>(
    db: &'db dyn Db,
    ty: Type<'db>,
    protocol_elements: &[Type<'db>],
) -> Option<Type<'db>> {
    if !all_elements_are_static(db, protocol_elements) {
        return None;
    }

    let Type::NominalInstance(instance) = ty else {
        return None;
    };

    let tuple = instance.own_tuple_spec(db)?;
    if !all_elements_are_static(db, tuple.all_elements()) {
        return None;
    }

    let protocol_tuple = TupleSpec::heterogeneous(protocol_elements.iter().copied());
    let Some(refined) = TupleSpecBuilder::from(tuple.as_ref()).intersect(db, &protocol_tuple)
    else {
        return Some(Type::Never);
    };

    Some(Type::tuple(TupleType::new(db, &refined.build())))
}

fn refine_tuple_protocol_intersection<'db>(
    db: &'db dyn Db,
    left: Type<'db>,
    right: Type<'db>,
) -> Option<Type<'db>> {
    let ((Type::ProtocolInstance(protocol), tuple) | (tuple, Type::ProtocolInstance(protocol))) =
        (left, right)
    else {
        return None;
    };
    let indexed = protocol.finite_indexed_constraint(db)?;
    refine_tuple_with_indexed_protocol(db, tuple, &indexed)
}

/// Plan subtraction of finite indexed constraints from a fixed-length tuple type.
///
/// The plan records only the tuple positions that differ in each alternative. This lets callers
/// count alternatives without constructing complete tuple types.
fn plan_indexed_protocol_complement<'db>(
    db: &'db dyn Db,
    ty: Type<'db>,
    protocol_elements: &[Type<'db>],
    max_alternatives: usize,
) -> TupleProtocolComplementPlan<'db> {
    if !all_elements_are_static(db, protocol_elements) {
        return TupleProtocolComplementPlan::NotApplicable;
    }

    let Type::NominalInstance(instance) = ty else {
        return TupleProtocolComplementPlan::NotApplicable;
    };
    let Some(tuple) = instance.own_tuple_spec(db) else {
        return TupleProtocolComplementPlan::NotApplicable;
    };
    let TupleSpec::Fixed(tuple) = tuple.as_ref() else {
        return TupleProtocolComplementPlan::NotApplicable;
    };
    if !all_elements_are_static(db, tuple.all_elements()) {
        return TupleProtocolComplementPlan::NotApplicable;
    }

    if tuple.len() != protocol_elements.len() {
        return TupleProtocolComplementPlan::Unchanged;
    }

    let mut remaining_elements = Vec::new();
    for (index, (element, protocol_element)) in tuple
        .all_elements()
        .iter()
        .zip(protocol_elements)
        .enumerate()
    {
        if element.is_disjoint_from(db, *protocol_element) {
            return TupleProtocolComplementPlan::Unchanged;
        }
        if element.is_subtype_of(db, *protocol_element) {
            continue;
        }

        let remaining_element =
            build_indexed_protocol_planning_intersection(db, [*element], [*protocol_element]);
        if remaining_element.is_never() {
            continue;
        }
        if remaining_elements.len() == max_alternatives {
            return TupleProtocolComplementPlan::ExceedsLimit;
        }
        remaining_elements.push((index, remaining_element));
    }

    if remaining_elements.is_empty() {
        TupleProtocolComplementPlan::Eliminated
    } else {
        TupleProtocolComplementPlan::Alternatives(remaining_elements)
    }
}

/// Try to merge a complementary guarded pair into an unguarded core.
///
/// e.g.
/// - `(A & ~AlwaysTruthy, A & ~AlwaysFalsy)` -> `Some(A)`
/// - `(A & ~AlwaysTruthy, B & ~AlwaysFalsy)` -> `Some(A | B)` if reconstruction is exact
/// - `(A & ~AlwaysTruthy, C)` -> `None`
///
/// Safety rule:
/// The candidate merge is accepted only if adding each original guard back reconstructs
/// exactly the original operands (`left` and `right`).
///
/// TODO: This processing is specialized for `AlwaysTruthy/AlwaysFalsy`.
/// It would be nice to generalize this in the future.
/// Discussion: <https://github.com/astral-sh/ty/issues/224>
fn merge_truthiness_guarded_pair<'db>(
    db: &'db dyn Db,
    left: Type<'db>,
    right: Type<'db>,
) -> Option<Type<'db>> {
    let (left_core, left_guard) = split_truthiness_guarded_intersection(db, left)?;
    let (right_core, right_guard) = split_truthiness_guarded_intersection(db, right)?;
    if left_guard == right_guard {
        return None;
    }

    if left_core.is_equivalent_to(db, right_core) {
        return Some(left_core);
    }

    let candidate = UnionType::from_elements(db, [left_core, right_core]);
    let left_reconstructed = IntersectionType::from_two_elements(db, candidate, left_guard);
    let right_reconstructed = IntersectionType::from_two_elements(db, candidate, right_guard);
    if left_reconstructed == left && right_reconstructed == right {
        Some(candidate)
    } else {
        None
    }
}

/// Combine union elements that cover more of the same enum class.
///
/// Enum complements are intersections like `Color & ~Literal[Color.RED]`. When a union contains
/// such a complement plus other complements or literals from the same enum, this rewrites the
/// element list to a single complement with the shared exclusions removed.
///
/// ```python
/// from enum import Enum
///
/// class Color(Enum):
///     RED = 1
///     BLUE = 2
///
/// # (Color excluding RED) | Literal[Color.RED] simplifies to Color.
/// ```
fn normalize_enum_complement_unions<'db>(db: &'db dyn Db, types: &mut Vec<Type<'db>>) -> bool {
    for complement_index in 0..types.len() {
        let Type::EnumComplement(complement) = types[complement_index] else {
            continue;
        };
        let enum_class = complement.enum_class(db);
        let metadata = enum_metadata(db, enum_class).expect("Enum complement class is an enum");
        let mut shared_excluded_names: FxHashSet<_> =
            complement.excluded_names(db).iter().cloned().collect();

        let mut remove_indices = Vec::new();
        for (index, ty) in types.iter().enumerate() {
            if index == complement_index {
                continue;
            }

            if let Type::EnumComplement(other_complement) = *ty {
                if other_complement.enum_class(db) == enum_class
                    && other_complement.rest(db) == complement.rest(db)
                {
                    shared_excluded_names
                        .retain(|name| other_complement.excluded_names(db).contains(name));
                    remove_indices.push(index);
                }
                continue;
            }

            if !complement.rest(db).is_empty() {
                continue;
            }

            let Some(enum_literal) = ty.as_enum_literal() else {
                continue;
            };
            if enum_literal.enum_class(db) != enum_class {
                continue;
            }

            let Some(canonical_name) = metadata.resolve_member(enum_literal.name(db)) else {
                continue;
            };
            shared_excluded_names.remove(canonical_name);
            remove_indices.push(index);
        }

        if !remove_indices.is_empty() {
            let mut builder =
                IntersectionBuilder::new(db).add_positive(enum_class.to_non_generic_instance(db));
            for rest in complement.rest(db) {
                builder = builder.add_positive(*rest);
            }
            for name in metadata
                .members
                .keys()
                .filter(|name| shared_excluded_names.contains(*name))
            {
                builder = builder.add_negative(Type::enum_literal(EnumLiteralType::new(
                    db,
                    enum_class,
                    name.clone(),
                )));
            }
            types[complement_index] = builder.build();

            remove_indices.sort_unstable();
            for index in remove_indices.into_iter().rev() {
                types.swap_remove(index);
            }
            return true;
        }
    }

    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiteralKind<'db> {
    Int,
    String,
    Bytes,
    Enum { enum_class: ClassLiteral<'db> },
}

impl<'db> Type<'db> {
    /// Return `true` if this type can be a supertype of some literals of `kind` and not others.
    fn splits_literals(self, db: &'db dyn Db, kind: LiteralKind) -> bool {
        match (self, kind) {
            // Note that as of 2026-01-04, `AlwaysFalsy` and `AlwaysTruthy` never split
            // enum literals, but that could change in the future. `Literal[Foo.X]` could
            // plausibly be understood by ty as a subtype of `AlwaysFalsy` in the following
            // snippet, because `Foo` is an IntEnum that does not override `__bool__` and
            // `Foo.X` has a falsy value whereas `Foo.Y` does not:
            //
            // ```py
            // class Foo(enum.IntEnum):
            //     X = 0
            //     Y = 1
            // ```
            (Type::AlwaysFalsy | Type::AlwaysTruthy, _) => true,
            (Type::LiteralValue(literal), _) => match (literal.kind(), kind) {
                (LiteralValueTypeKind::String(_), LiteralKind::String) => true,
                (LiteralValueTypeKind::Bytes(_), LiteralKind::Bytes) => true,
                (LiteralValueTypeKind::Int(_), LiteralKind::Int) => true,
                (LiteralValueTypeKind::Enum(enum_literal), LiteralKind::Enum { enum_class }) => {
                    enum_literal.enum_class(db) == enum_class
                }
                _ => false,
            },
            (Type::Intersection(intersection), _) => {
                intersection
                    .positive(db)
                    .iter()
                    .any(|ty| ty.splits_literals(db, kind))
                    || intersection
                        .negative(db)
                        .iter()
                        .any(|ty| ty.splits_literals(db, kind))
            }
            (Type::Union(union), _) => union
                .elements(db)
                .iter()
                .any(|ty| ty.splits_literals(db, kind)),
            (Type::EnumComplement(complement), LiteralKind::Enum { enum_class }) => {
                complement.enum_class(db) == enum_class
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
enum UnionElement<'db> {
    Type(Type<'db>),
    // A map from integer literals to their promotability.
    //
    // Note that an unpromotable literal takes higher precedence than the identical literal
    // in its promotable form.
    IntLiterals(FxOrderMap<i64, bool>),
    StringLiterals(FxOrderMap<StringLiteralType<'db>, bool>),
    BytesLiterals(FxOrderMap<BytesLiteralType<'db>, bool>),
    EnumLiterals {
        enum_class: ClassLiteral<'db>,
        literals: FxOrderMap<EnumLiteralType<'db>, bool>,
    },
}

impl<'db> UnionElement<'db> {
    fn type_count(&self) -> usize {
        match self {
            UnionElement::Type(_) => 1,
            UnionElement::IntLiterals(literals) => literals.len(),
            UnionElement::StringLiterals(literals) => literals.len(),
            UnionElement::BytesLiterals(literals) => literals.len(),
            UnionElement::EnumLiterals { literals, .. } => literals.len(),
        }
    }

    /// Try reducing this `UnionElement` given the presence in the same union of `other_type`.
    fn try_reduce(&mut self, db: &'db dyn Db, other_type: Type<'db>) -> ReduceResult<'db> {
        let mut other_type_negated_cache = None;
        let mut other_type_negated =
            || *other_type_negated_cache.get_or_insert_with(|| other_type.negate(db));

        let mut collapse = false;
        let mut ignore = false;

        // A closure called for each element in a set of literals
        // to determine whether the element should be retained in the set.
        //
        // If `ignore` or `collapse` is `true` for any element in the set,
        // we no longer need to do any expensive redundancy checks for any
        // further elements in the set:
        //
        // - if `ignore` is `true`, this indicates that `other_type` is
        //   redundant with one of the literals in this set. Given this fact,
        //   it cannot be possible for any other literals in this set to be
        //   redundant with `other_type`.
        // - if `collapse` is `true`, all literals of this kind will be
        //   removed from the union, so it's irrelevant to answer the
        //   question of which literals should remain in this set.
        //
        // We therefore only ask if `ty` is redundant with `other_type` if
        // both `ignore` and `collapse` are `false`. If either is `true`,
        // we skip the expensive redundancy check and return `true`.
        let mut should_retain_type = |ty| {
            if ignore || other_type.is_redundant_with(db, ty) {
                ignore = true;
                return true;
            }
            if collapse || other_type_negated().is_subtype_of(db, ty) {
                collapse = true;
                return true;
            }
            !ty.is_redundant_with(db, other_type)
        };

        let should_keep = match self {
            UnionElement::IntLiterals(literals) => {
                if other_type.splits_literals(db, LiteralKind::Int) {
                    literals.retain(|literal, promotable| {
                        should_retain_type(LiteralValueType::new(*literal, *promotable).into())
                    });
                    !literals.is_empty()
                } else {
                    let (literal, promotable) = literals.first().unwrap();
                    !Type::from(LiteralValueType::new(*literal, *promotable))
                        .is_redundant_with(db, other_type)
                }
            }
            UnionElement::StringLiterals(literals) => {
                if other_type.splits_literals(db, LiteralKind::String) {
                    literals.retain(|literal, promotable| {
                        should_retain_type(LiteralValueType::new(*literal, *promotable).into())
                    });
                    !literals.is_empty()
                } else {
                    let (literal, promotable) = literals.first().unwrap();
                    !Type::from(LiteralValueType::new(*literal, *promotable))
                        .is_redundant_with(db, other_type)
                }
            }
            UnionElement::BytesLiterals(literals) => {
                if other_type.splits_literals(db, LiteralKind::Bytes) {
                    literals.retain(|literal, promotable| {
                        should_retain_type(LiteralValueType::new(*literal, *promotable).into())
                    });
                    !literals.is_empty()
                } else {
                    let (literal, promotable) = literals.first().unwrap();
                    !Type::from(LiteralValueType::new(*literal, *promotable))
                        .is_redundant_with(db, other_type)
                }
            }
            UnionElement::EnumLiterals {
                enum_class,
                literals,
            } => {
                let enum_class = LiteralKind::Enum {
                    enum_class: *enum_class,
                };
                if other_type.splits_literals(db, enum_class) {
                    literals.retain(|literal, promotable| {
                        should_retain_type(LiteralValueType::new(*literal, *promotable).into())
                    });
                    !literals.is_empty()
                } else {
                    let (literal, promotable) = literals.first().unwrap();
                    !Type::from(LiteralValueType::new(*literal, *promotable))
                        .is_redundant_with(db, other_type)
                }
            }
            UnionElement::Type(existing) => return ReduceResult::Type(*existing),
        };

        if ignore {
            ReduceResult::Ignore
        } else if collapse {
            ReduceResult::CollapseToObject
        } else {
            ReduceResult::KeepIf(should_keep)
        }
    }
}

enum ReduceResult<'db> {
    /// Reduction of this `UnionElement` is complete; keep it in the union if the nested
    /// boolean is true, eliminate it from the union if false.
    KeepIf(bool),
    /// Collapse this entire union to `object`.
    CollapseToObject,
    /// The new element is a subtype of an existing part of the `UnionElement`, ignore it.
    Ignore,
    /// The given `Type` can stand-in for the entire `UnionElement` for further union
    /// simplification checks.
    Type(Type<'db>),
}

/// If the value ​​is defined recursively, widening is performed from fewer literal elements,
/// resulting in faster convergence of the fixed-point iteration.
const MAX_RECURSIVE_UNION_LITERALS: usize = 5;
/// If the value ​​is defined non-recursively, the fixed-point iteration will converge in one go,
/// so in principle we can have as many literal elements as we want,
/// but to avoid unintended huge computational loads, we limit it to 256.
const MAX_NON_RECURSIVE_UNION_LITERALS: usize = 256;
/// However, we set a much larger limit for enum literals than for other kinds of literals.
/// Huge enums are not uncommon (especially in generated code), and it's annoying
/// if reachability analysis etc. fails when analysing these enums.
const MAX_NON_RECURSIVE_UNION_ENUM_LITERALS: usize = 8192;
pub(crate) struct UnionBuilder<'db> {
    elements: Vec<UnionElement<'db>>,
    db: &'db dyn Db,
    unpack_aliases: bool,
    /// This is enabled when joining types in a `cycle_recovery` function.
    /// Since a cycle cannot be created within a `cycle_recovery` function,
    /// execution of `is_redundant_with` is skipped.
    cycle_recovery: bool,
    recursively_defined: RecursivelyDefined,
}

/// Accumulates types into a union.
///
/// Most real-world type variables only accumulate one or two constraints. We keep those cases as
/// plain `Type`s and only allocate a `UnionBuilder` once we know the accumulation is larger.
pub(crate) enum UnionAccumulator<'db> {
    One(Type<'db>),
    Two(Type<'db>, Type<'db>),
    Deferred(UnionBuilder<'db>),
}

impl<'db> UnionAccumulator<'db> {
    pub(crate) fn new(ty: Type<'db>) -> Self {
        UnionAccumulator::One(ty)
    }

    pub(crate) fn add(&mut self, db: &'db dyn Db, ty: Type<'db>) {
        match self {
            UnionAccumulator::One(existing) => {
                *self = UnionAccumulator::Two(*existing, ty);
            }
            UnionAccumulator::Two(first, second) => {
                let mut builder = UnionBuilder::new(db);
                builder.add_in_place(*first);
                builder.add_in_place(*second);
                builder.add_in_place(ty);
                *self = UnionAccumulator::Deferred(builder);
            }
            UnionAccumulator::Deferred(builder) => builder.add_in_place(ty),
        }
    }

    pub(crate) fn get_or_build(&mut self, db: &'db dyn Db) -> Type<'db> {
        match self {
            UnionAccumulator::One(ty) => *ty,
            UnionAccumulator::Two(first, second) => {
                let ty = UnionType::from_two_elements(db, *first, *second);
                *self = UnionAccumulator::One(ty);
                ty
            }
            UnionAccumulator::Deferred(_) => {
                let ty = std::mem::replace(self, UnionAccumulator::new(Type::Never)).into_type(db);
                *self = UnionAccumulator::new(ty);
                ty
            }
        }
    }

    pub(crate) fn into_type(self, db: &'db dyn Db) -> Type<'db> {
        match self {
            UnionAccumulator::One(ty) => ty,
            UnionAccumulator::Two(first, second) => UnionType::from_two_elements(db, first, second),
            UnionAccumulator::Deferred(builder) => builder.build(),
        }
    }
}

impl<'db> UnionBuilder<'db> {
    pub(crate) fn new(db: &'db dyn Db) -> Self {
        Self {
            db,
            elements: vec![],
            unpack_aliases: true,
            cycle_recovery: false,
            recursively_defined: RecursivelyDefined::No,
        }
    }

    pub(crate) fn unpack_aliases(mut self, val: bool) -> Self {
        self.unpack_aliases = val;
        self
    }

    pub(crate) fn cycle_recovery(mut self, val: bool) -> Self {
        self.cycle_recovery = val;
        if self.cycle_recovery {
            self.unpack_aliases = false;
        }
        self
    }

    pub(crate) fn recursively_defined(mut self, val: RecursivelyDefined) -> Self {
        self.recursively_defined = val;
        self
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Collapse the union to a single type: `object`.
    fn collapse_to_object(&mut self) {
        self.elements.clear();
        self.elements.push(UnionElement::Type(Type::object()));
    }

    fn widen_literal_types(&mut self, seen_aliases: &mut Vec<Type<'db>>) {
        let mut replace_with = vec![];
        for elem in &self.elements {
            match elem {
                UnionElement::IntLiterals(_) => {
                    replace_with.push(KnownClass::Int.to_instance(self.db));
                }
                UnionElement::StringLiterals(_) => {
                    replace_with.push(KnownClass::Str.to_instance(self.db));
                }
                UnionElement::BytesLiterals(_) => {
                    replace_with.push(KnownClass::Bytes.to_instance(self.db));
                }
                UnionElement::EnumLiterals { literals, .. } => {
                    let (enum_literal, _) = literals.first().unwrap();
                    replace_with.push(enum_literal.enum_class_instance(self.db));
                }
                UnionElement::Type(_) => {}
            }
        }
        for ty in replace_with {
            self.add_in_place_impl(ty, seen_aliases);
        }
    }

    /// Adds a type to this union.
    pub(crate) fn add(mut self, ty: Type<'db>) -> Self {
        self.add_in_place(ty);
        self
    }

    /// Adds a type to this union.
    pub(crate) fn add_in_place(&mut self, ty: Type<'db>) {
        self.add_in_place_impl(ty, &mut vec![]);
    }

    pub(crate) fn add_in_place_impl(&mut self, ty: Type<'db>, seen_aliases: &mut Vec<Type<'db>>) {
        let cycle_recovery = self.cycle_recovery;
        let should_widen = |literals, recursively_defined: RecursivelyDefined| {
            if recursively_defined.is_yes() && cycle_recovery {
                literals >= MAX_RECURSIVE_UNION_LITERALS
            } else {
                literals >= MAX_NON_RECURSIVE_UNION_LITERALS
            }
        };

        let mut ty_negated_cache = None;
        let mut ty_negated = || *ty_negated_cache.get_or_insert_with(|| ty.negate(self.db));

        match ty {
            Type::Union(union) => {
                let new_elements = union.elements(self.db);
                self.elements.reserve(new_elements.len());
                for element in new_elements {
                    self.add_in_place_impl(*element, seen_aliases);
                }
                self.recursively_defined = self
                    .recursively_defined
                    .or(union.recursively_defined(self.db));
                if self.cycle_recovery && self.recursively_defined.is_yes() {
                    let literals = self.elements.iter().fold(0, |acc, elem| match elem {
                        UnionElement::IntLiterals(literals) => acc + literals.len(),
                        UnionElement::StringLiterals(literals) => acc + literals.len(),
                        UnionElement::BytesLiterals(literals) => acc + literals.len(),
                        UnionElement::EnumLiterals { literals, .. } => acc + literals.len(),
                        UnionElement::Type(_) => acc,
                    });
                    if should_widen(literals, self.recursively_defined) {
                        self.widen_literal_types(seen_aliases);
                    }
                }
            }
            // Adding `Never` to a union is a no-op.
            Type::Never => {}
            Type::TypeAlias(alias) if self.unpack_aliases => {
                if seen_aliases.contains(&ty) {
                    // Union contains itself recursively via a type alias. This is an error, just
                    // leave out the recursive alias. TODO surface this error.
                } else {
                    seen_aliases.push(ty);
                    self.add_in_place_impl(alias.value_type(self.db), seen_aliases);
                }
            }
            Type::LiteralValue(literal) => {
                self.recursively_defined =
                    self.recursively_defined.or(literal.recursively_defined());
                match literal.kind() {
                    // If adding a string literal, look for an existing `UnionElement::StringLiterals` to
                    // add it to, or an existing element that is a super-type of string literals, which
                    // means we shouldn't add it. Otherwise, add a new `UnionElement::StringLiterals`
                    // containing it.
                    LiteralValueTypeKind::String(string_literal) => {
                        let mut found = None;
                        let mut to_remove = None;
                        for (index, element) in self.elements.iter_mut().enumerate() {
                            match element {
                                UnionElement::StringLiterals(literals) => {
                                    if should_widen(literals.len(), self.recursively_defined) {
                                        let replace_with = KnownClass::Str.to_instance(self.db);
                                        self.add_in_place_impl(replace_with, seen_aliases);
                                        return;
                                    }
                                    found = Some(literals);
                                    continue;
                                }
                                UnionElement::Type(existing) => {
                                    // e.g. `existing` could be `Literal[""] & Any`,
                                    // and `ty` could be `Literal[""]`
                                    if ty.is_redundant_with(self.db, *existing) {
                                        return;
                                    }
                                    if existing.is_redundant_with(self.db, ty) {
                                        to_remove = Some(index);
                                        continue;
                                    }
                                    if ty_negated().is_subtype_of(self.db, *existing) {
                                        // The type that includes both this new element, and its negation
                                        // (or a supertype of its negation), must be simply `object`.
                                        self.collapse_to_object();
                                        return;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if let Some(found) = found {
                            let is_promotable = literal.is_promotable();
                            *found.entry(string_literal).or_insert(is_promotable) &= is_promotable;
                        } else {
                            self.elements.push(UnionElement::StringLiterals(
                                FxOrderMap::from_iter([(string_literal, literal.is_promotable())]),
                            ));
                        }
                        if let Some(index) = to_remove {
                            self.elements.swap_remove(index);
                        }
                    }
                    // Same for bytes literals as for string literals, above.
                    LiteralValueTypeKind::Bytes(bytes_literal) => {
                        let mut found = None;
                        let mut to_remove = None;
                        for (index, element) in self.elements.iter_mut().enumerate() {
                            match element {
                                UnionElement::BytesLiterals(literals) => {
                                    if should_widen(literals.len(), self.recursively_defined) {
                                        let replace_with = KnownClass::Bytes.to_instance(self.db);
                                        self.add_in_place_impl(replace_with, seen_aliases);
                                        return;
                                    }
                                    found = Some(literals);
                                    continue;
                                }
                                UnionElement::Type(existing) => {
                                    if ty.is_redundant_with(self.db, *existing) {
                                        return;
                                    }
                                    // e.g. `existing` could be `Literal[b""] & Any`,
                                    // and `ty` could be `Literal[b""]`
                                    if existing.is_redundant_with(self.db, ty) {
                                        to_remove = Some(index);
                                        continue;
                                    }
                                    if ty_negated().is_subtype_of(self.db, *existing) {
                                        // The type that includes both this new element, and its negation
                                        // (or a supertype of its negation), must be simply `object`.
                                        self.collapse_to_object();
                                        return;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if let Some(found) = found {
                            let is_promotable = literal.is_promotable();
                            *found.entry(bytes_literal).or_insert(is_promotable) &= is_promotable;
                        } else {
                            self.elements
                                .push(UnionElement::BytesLiterals(FxOrderMap::from_iter([(
                                    bytes_literal,
                                    literal.is_promotable(),
                                )])));
                        }
                        if let Some(index) = to_remove {
                            self.elements.swap_remove(index);
                        }
                    }
                    // And same for int literals as well.
                    LiteralValueTypeKind::Int(int_literal) => {
                        let mut found = None;
                        let mut to_remove = None;
                        for (index, element) in self.elements.iter_mut().enumerate() {
                            match element {
                                UnionElement::IntLiterals(literals) => {
                                    if should_widen(literals.len(), self.recursively_defined) {
                                        let replace_with = KnownClass::Int.to_instance(self.db);
                                        self.add_in_place_impl(replace_with, seen_aliases);
                                        return;
                                    }
                                    found = Some(literals);
                                    continue;
                                }
                                UnionElement::Type(existing) => {
                                    if ty.is_redundant_with(self.db, *existing) {
                                        return;
                                    }
                                    // e.g. `existing` could be `Literal[1] & Any`,
                                    // and `ty` could be `Literal[1]`
                                    if existing.is_redundant_with(self.db, ty) {
                                        to_remove = Some(index);
                                        continue;
                                    }
                                    if ty_negated().is_subtype_of(self.db, *existing) {
                                        // The type that includes both this new element, and its negation
                                        // (or a supertype of its negation), must be simply `object`.
                                        self.collapse_to_object();
                                        return;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if let Some(found) = found {
                            let is_promotable = literal.is_promotable();
                            *found.entry(int_literal.as_i64()).or_insert(is_promotable) &=
                                is_promotable;
                        } else {
                            self.elements
                                .push(UnionElement::IntLiterals(FxOrderMap::from_iter([(
                                    int_literal.as_i64(),
                                    literal.is_promotable(),
                                )])));
                        }
                        if let Some(index) = to_remove {
                            self.elements.swap_remove(index);
                        }
                    }
                    LiteralValueTypeKind::Enum(enum_member_to_add) => {
                        let enum_class = enum_member_to_add.enum_class(self.db);

                        // We generally expect that a `Type::LiteralValue(LiteralValueTypeKind::Enum)`
                        // value is in fact in enum, i.e., that `enum_metadata` returns `Some(...)`.
                        // However, during cycle recovery, it's possible (empirically) to end up
                        // in an inconsistent state. The metadata is only required for simplification
                        // and not for correctness, so we treat it as optional here.
                        // TODO: Come up with a design, either to enum metadata or the cycle
                        // handling more broadly, that avoids this inconsistency.
                        let metadata = enum_metadata(self.db, enum_class);

                        if metadata.is_some_and(|metadata| metadata.members.len() == 1) {
                            self.add_in_place_impl(
                                enum_member_to_add.enum_class_instance(self.db),
                                seen_aliases,
                            );
                            return;
                        }

                        let mut found = None;
                        let mut to_remove = None;
                        for (index, element) in self.elements.iter_mut().enumerate() {
                            match element {
                                UnionElement::EnumLiterals {
                                    enum_class: existing_enum_class,
                                    literals,
                                } => {
                                    if *existing_enum_class != enum_class {
                                        continue;
                                    }
                                    // See the doc-comment above `MAX_NON_RECURSIVE_UNION_ENUM_LITERALS`
                                    // for why we avoid using the `should_widen` closure here.
                                    let enum_literals_limit =
                                        if self.recursively_defined.is_yes() && cycle_recovery {
                                            MAX_RECURSIVE_UNION_LITERALS
                                        } else {
                                            MAX_NON_RECURSIVE_UNION_ENUM_LITERALS
                                        };
                                    if literals.len() >= enum_literals_limit {
                                        let (literal, _) = literals.first().unwrap();
                                        let replace_with = literal.enum_class_instance(self.db);
                                        self.add_in_place_impl(replace_with, seen_aliases);
                                        return;
                                    }
                                    found = Some(literals);
                                    continue;
                                }
                                UnionElement::Type(existing) => {
                                    if ty.is_redundant_with(self.db, *existing) {
                                        return;
                                    }
                                    // e.g. `existing` could be `Literal[Foo.X] & Any`,
                                    // and `ty` could be `Literal[Foo.X]`
                                    if existing.is_redundant_with(self.db, ty) {
                                        to_remove = Some(index);
                                        continue;
                                    }
                                    if ty_negated().is_subtype_of(self.db, *existing) {
                                        // The type that includes both this new element, and its negation
                                        // (or a supertype of its negation), must be simply `object`.
                                        self.collapse_to_object();
                                        return;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if let Some(found) = found {
                            match found.entry(enum_member_to_add) {
                                ordermap::map::Entry::Vacant(entry) => {
                                    entry.insert(literal.is_promotable());

                                    if metadata.is_some_and(|metadata| {
                                        found.len() == metadata.members.len()
                                    }) {
                                        self.add_in_place_impl(
                                            enum_member_to_add.enum_class_instance(self.db),
                                            seen_aliases,
                                        );
                                        return;
                                    }
                                }
                                ordermap::map::Entry::Occupied(mut entry) => {
                                    *entry.get_mut() &= literal.is_promotable();
                                }
                            }
                        } else {
                            self.elements.push(UnionElement::EnumLiterals {
                                enum_class,
                                literals: FxOrderMap::from_iter([(
                                    enum_member_to_add,
                                    literal.is_promotable(),
                                )]),
                            });
                        }
                        if let Some(index) = to_remove {
                            self.elements.swap_remove(index);
                        }
                    }
                    _ => self.push_type(ty, seen_aliases),
                }
            }
            // Adding `object` to a union results in `object`.
            ty if ty.is_object() => self.collapse_to_object(),
            _ => self.push_type(ty, seen_aliases),
        }
    }

    fn push_type(&mut self, ty: Type<'db>, seen_aliases: &mut Vec<Type<'db>>) {
        let mut ty = ty;
        let bool_pair = |ty: Type<'db>| {
            if let Some(LiteralValueTypeKind::Bool(b)) = ty.as_literal_value_kind() {
                Some(LiteralValueTypeKind::Bool(!b))
            } else {
                None
            }
        };

        // If an alias gets here, it means we aren't unpacking aliases, and we also
        // shouldn't try to simplify aliases out of the union, because that will require
        // unpacking them.
        let should_simplify_full = !matches!(ty, Type::TypeAlias(_)) && !self.cycle_recovery;

        let mut ty_negated: Option<Type> = None;
        let mut to_remove = SmallVec::<[usize; 2]>::new();

        for (i, element) in self.elements.iter_mut().enumerate() {
            let element_type = match element.try_reduce(self.db, ty) {
                ReduceResult::KeepIf(keep) => {
                    if !keep {
                        to_remove.push(i);
                    }
                    continue;
                }
                ReduceResult::Type(ty) => ty,
                ReduceResult::CollapseToObject => {
                    self.collapse_to_object();
                    return;
                }
                ReduceResult::Ignore => {
                    return;
                }
            };

            if ty == element_type {
                return;
            }

            // Fold `(T & ~AlwaysTruthy) | (T & ~AlwaysFalsy)` to `T`.
            if let Some(merged_type) = merge_truthiness_guarded_pair(self.db, ty, element_type) {
                to_remove.push(i);
                ty = merged_type;
                continue;
            }

            if element_type
                .as_literal_value_kind()
                .zip(bool_pair(ty))
                .is_some_and(|(element, pair)| element == pair)
            {
                self.add_in_place_impl(KnownClass::Bool.to_instance(self.db), seen_aliases);
                return;
            }

            // Comparing `TypedDict`s for redundancy requires iterating over their fields, which is
            // problematic if some of those fields point to recursive `Union`s. To avoid cycles,
            // compare `TypedDict`s by name/identity instead of using the `has_relation_to`
            // machinery.
            if element_type.is_typed_dict() && ty.is_typed_dict() {
                continue;
            }

            if should_simplify_full && !matches!(element_type, Type::TypeAlias(_)) {
                if ty.is_redundant_with(self.db, element_type) {
                    return;
                }

                if element_type.is_redundant_with(self.db, ty) {
                    to_remove.push(i);
                    continue;
                }

                let negated = ty_negated.get_or_insert_with(|| ty.negate(self.db));
                if negated.is_subtype_of(self.db, element_type) {
                    // We add `ty` to the union. We just checked that `~ty` is a subtype of an
                    // existing `element`. This also means that `~ty | ty` is a subtype of
                    // `element | ty`, because both elements in the first union are subtypes of
                    // the corresponding elements in the second union. But `~ty | ty` is just
                    // `object`. Since `object` is a subtype of `element | ty`, we can only
                    // conclude that `element | ty` must be `object` (object has no other
                    // supertypes). This means we can simplify the whole union to just
                    // `object`, since all other potential elements would also be subtypes of
                    // `object`.
                    self.collapse_to_object();
                    return;
                }
            }
        }

        let mut to_remove = to_remove.into_iter();
        if let Some(first) = to_remove.next() {
            self.elements[first] = UnionElement::Type(ty);
            // We iterate in descending order to keep remaining indices valid after `swap_remove`.
            for index in to_remove.rev() {
                self.elements.swap_remove(index);
            }
        } else {
            self.elements.push(UnionElement::Type(ty));
        }
    }

    pub(crate) fn build(self) -> Type<'db> {
        self.try_build().unwrap_or(Type::Never)
    }

    pub(crate) fn try_build(self) -> Option<Type<'db>> {
        let db = self.db;
        let unpack_aliases = self.unpack_aliases;
        let cycle_recovery = self.cycle_recovery;
        let recursively_defined = self.recursively_defined;

        let type_count = self.elements.iter().map(UnionElement::type_count).sum();
        let mut types = Vec::with_capacity(type_count);
        for element in self.elements {
            match element {
                UnionElement::IntLiterals(literals) => {
                    types.extend(literals.into_iter().map(|(literal, promotable)| {
                        Type::from(
                            LiteralValueType::new(literal, promotable)
                                .with_recursively_defined(recursively_defined),
                        )
                    }));
                }
                UnionElement::StringLiterals(literals) => {
                    types.extend(literals.into_iter().map(|(literal, promotable)| {
                        Type::from(
                            LiteralValueType::new(literal, promotable)
                                .with_recursively_defined(recursively_defined),
                        )
                    }));
                }
                UnionElement::BytesLiterals(literals) => {
                    types.extend(literals.into_iter().map(|(literal, promotable)| {
                        Type::from(
                            LiteralValueType::new(literal, promotable)
                                .with_recursively_defined(recursively_defined),
                        )
                    }));
                }
                UnionElement::EnumLiterals { literals, .. } => {
                    types.extend(literals.into_iter().map(|(literal, promotable)| {
                        Type::from(
                            LiteralValueType::new(literal, promotable)
                                .with_recursively_defined(recursively_defined),
                        )
                    }));
                }
                UnionElement::Type(ty) => types.push(ty),
            }
        }

        if normalize_enum_complement_unions(db, &mut types) {
            let builder = UnionBuilder::new(db)
                .unpack_aliases(unpack_aliases)
                .cycle_recovery(cycle_recovery)
                .recursively_defined(recursively_defined);
            return types
                .into_iter()
                .fold(builder, UnionBuilder::add)
                .try_build();
        }

        match types.len() {
            0 => None,
            1 => Some(types[0]),
            _ => Some(Type::Union(UnionType::new(
                db,
                types.into_boxed_slice(),
                recursively_defined,
            ))),
        }
    }
}

#[derive(Clone)]
pub(crate) struct IntersectionBuilder<'db> {
    // Really this builds a union-of-intersections, because we always keep our set-theoretic types
    // in disjunctive normal form (DNF), a union of intersections. In the simplest case there's
    // just a single intersection in this vector, and we are building a single intersection type,
    // but if a union is added to the intersection, we'll distribute ourselves over that union and
    // create a union of intersections.
    intersections: Vec<InnerIntersectionBuilder<'db>>,
    db: &'db dyn Db,
}

impl<'db> IntersectionBuilder<'db> {
    pub(crate) fn new(db: &'db dyn Db) -> Self {
        Self {
            db,
            intersections: vec![InnerIntersectionBuilder::default()],
        }
    }

    fn empty(db: &'db dyn Db) -> Self {
        Self {
            db,
            intersections: vec![],
        }
    }

    pub(crate) fn add_positive(self, ty: Type<'db>) -> Self {
        self.add_positive_impl(ty, &mut vec![])
    }

    pub(crate) fn add_positive_impl(
        mut self,
        ty: Type<'db>,
        seen_aliases: &mut Vec<Type<'db>>,
    ) -> Self {
        match ty {
            Type::TypeAlias(alias) => {
                if seen_aliases.contains(&ty) {
                    // Recursive alias, add it without expanding to avoid infinite recursion.
                    for inner in &mut self.intersections {
                        inner.positive.insert(ty);
                    }
                    return self;
                }
                seen_aliases.push(ty);
                let value_type = alias.value_type(self.db);
                self.add_positive_impl(value_type, seen_aliases)
            }
            Type::Union(union) => {
                // Distribute ourself over this union: for each union element, clone ourself and
                // intersect with that union element, then create a new union-of-intersections with all
                // of those sub-intersections in it. E.g. if `self` is a simple intersection `T1 & T2`
                // and we add `T3 | T4` to the intersection, we don't get `T1 & T2 & (T3 | T4)` (that's
                // not in DNF), we distribute the union and get `(T1 & T3) | (T2 & T3) | (T1 & T4) |
                // (T2 & T4)`. If `self` is already a union-of-intersections `(T1 & T2) | (T3 & T4)`
                // and we add `T5 | T6` to it, that flattens all the way out to `(T1 & T2 & T5) | (T1 &
                // T2 & T6) | (T3 & T4 & T5) ...` -- you get the idea.
                union
                    .elements(self.db)
                    .iter()
                    .map(|elem| self.clone().add_positive_impl(*elem, seen_aliases))
                    .fold(IntersectionBuilder::empty(self.db), |mut builder, sub| {
                        builder.intersections.extend(sub.intersections);
                        builder
                    })
            }
            // `(A & B & ~C) & (D & E & ~F)` -> `A & B & D & E & ~C & ~F`
            Type::Intersection(other) => {
                let db = self.db;
                for pos in other.positive(db) {
                    self = self.add_positive_impl(*pos, seen_aliases);
                }
                for neg in other.negative(db) {
                    self = self.add_negative_impl(*neg, seen_aliases);
                }
                self
            }
            Type::EnumComplement(complement) => {
                let db = self.db;
                self.add_positive_impl(complement.to_intersection(db), seen_aliases)
            }
            _ => {
                // If we are already a union-of-intersections, distribute the new intersected element
                // across all of those intersections.
                for inner in &mut self.intersections {
                    inner.add_positive(self.db, ty);
                }
                self
            }
        }
    }

    pub(crate) fn add_negative(self, ty: Type<'db>) -> Self {
        self.add_negative_impl(ty, &mut vec![])
    }

    pub(crate) fn add_negative_impl(
        mut self,
        ty: Type<'db>,
        seen_aliases: &mut Vec<Type<'db>>,
    ) -> Self {
        // See comments above in `add_positive`; this is just the negated version.
        match ty {
            Type::TypeAlias(alias) => {
                if seen_aliases.contains(&ty) {
                    // Recursive alias, add it without expanding to avoid infinite recursion.
                    for inner in &mut self.intersections {
                        inner.negative.insert(ty);
                    }
                    return self;
                }
                seen_aliases.push(ty);
                let value_type = alias.value_type(self.db);
                self.add_negative_impl(value_type, seen_aliases)
            }
            Type::Union(union) => {
                for elem in union.elements(self.db) {
                    self = self.add_negative_impl(*elem, seen_aliases);
                }
                self
            }
            Type::Intersection(intersection) => {
                // (A | B) & ~(C & ~D)
                // -> (A | B) & (~C | D)
                // -> ((A | B) & ~C) | ((A | B) & D)
                // i.e. if we have an intersection of positive constraints C
                // and negative constraints D, then our new intersection
                // is (existing & ~C) | (existing & D)

                let positive_side = intersection
                    .positive(self.db)
                    .iter()
                    // we negate all the positive constraints while distributing
                    .map(|elem| {
                        self.clone()
                            .add_negative_impl(*elem, &mut seen_aliases.clone())
                    });

                let negative_side = intersection
                    .negative(self.db)
                    .iter()
                    // all negative constraints end up becoming positive constraints
                    .map(|elem| {
                        self.clone()
                            .add_positive_impl(*elem, &mut seen_aliases.clone())
                    });

                positive_side.chain(negative_side).fold(
                    IntersectionBuilder::empty(self.db),
                    |mut builder, sub| {
                        builder.intersections.extend(sub.intersections);
                        builder
                    },
                )
            }
            Type::EnumComplement(complement) => {
                let db = self.db;
                self.add_negative_impl(complement.to_intersection(db), seen_aliases)
            }
            _ => {
                for inner in &mut self.intersections {
                    inner.add_negative(self.db, ty);
                }
                self
            }
        }
    }

    fn distribute_indexed_protocol_negatives(self) -> Self {
        let protocols = self.indexed_protocol_negatives();
        if protocols.is_empty() {
            return self;
        }
        let Ok(branch_plans) = self.plan_indexed_protocol_complements(&protocols) else {
            return self;
        };

        let db = self.db;
        Self {
            db,
            intersections: self
                .intersections
                .into_iter()
                .zip(branch_plans)
                .flat_map(|(inner, branch_plan)| {
                    if let Some(branch_plan) = branch_plan {
                        inner.materialize_indexed_protocol_complements(db, branch_plan)
                    } else {
                        vec![inner]
                    }
                })
                .collect(),
        }
    }

    fn indexed_protocol_negatives(&self) -> FxOrderSet<Type<'db>> {
        self.intersections
            .iter()
            .flat_map(|inner| inner.negative.iter())
            .filter(|negative| {
                let Type::ProtocolInstance(protocol) = negative else {
                    return false;
                };
                protocol.finite_indexed_constraint(self.db).is_some()
            })
            .copied()
            .collect()
    }

    fn plan_indexed_protocol_complements(
        &self,
        protocols: &FxOrderSet<Type<'db>>,
    ) -> Result<Vec<Option<InnerIndexedProtocolComplementsPlan<'db>>>, ()> {
        let mut total_alternatives = 0usize;
        let mut total_materializations = 0usize;
        let mut branches = Vec::with_capacity(self.intersections.len());
        for inner in &self.intersections {
            if inner.positive.contains(&Type::Never) {
                branches.push(None);
                continue;
            }
            let plan = inner.plan_indexed_protocol_complements(
                self.db,
                protocols,
                MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES,
            )?;
            total_alternatives = total_alternatives
                .saturating_add(plan.as_ref().map_or(1, |plan| plan.alternatives.len()));
            total_materializations =
                total_materializations.saturating_add(plan.as_ref().map_or(0, |plan| {
                    plan.alternatives.iter().map(Vec::len).sum::<usize>()
                }));
            if total_alternatives > MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES
                || total_materializations > MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES
            {
                return Err(());
            }
            branches.push(plan);
        }
        Ok(branches)
    }

    pub(crate) fn positive_elements<I, T>(mut self, elements: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Type<'db>>,
    {
        for element in elements {
            self = self.add_positive(element.into());
        }
        self
    }

    pub(crate) fn build(self) -> Type<'db> {
        self.distribute_indexed_protocol_negatives()
            .build_without_indexed_protocol_distribution()
    }

    fn build_without_indexed_protocol_distribution(self) -> Type<'db> {
        UnionType::from_elements(
            self.db,
            self.intersections
                .into_iter()
                .map(|inner| inner.build(self.db)),
        )
    }
}

#[derive(Debug, Clone, Default)]
struct InnerIntersectionBuilder<'db> {
    positive: FxOrderSet<Type<'db>>,
    negative: NegativeIntersectionElements<'db>,
}

impl<'db> InnerIntersectionBuilder<'db> {
    fn plan_indexed_protocol_complement(
        &self,
        db: &'db dyn Db,
        protocol_elements: &[Type<'db>],
        max_alternatives: usize,
    ) -> Result<Option<InnerIndexedProtocolComplementPlan<'db>>, ()> {
        let mut unchanged = false;
        let mut exceeded_limit = false;
        let mut best_alternative_count = usize::MAX;
        let mut best_alternatives = Vec::new();

        for (positive_index, existing_positive) in self.positive.iter().enumerate() {
            match plan_indexed_protocol_complement(
                db,
                *existing_positive,
                protocol_elements,
                max_alternatives,
            ) {
                TupleProtocolComplementPlan::NotApplicable => continue,
                TupleProtocolComplementPlan::ExceedsLimit => exceeded_limit = true,
                TupleProtocolComplementPlan::Unchanged => unchanged = true,
                TupleProtocolComplementPlan::Eliminated => {
                    return Ok(Some(InnerIndexedProtocolComplementPlan::Eliminated));
                }
                TupleProtocolComplementPlan::Alternatives(tuple_changes) => {
                    match tuple_changes.len().cmp(&best_alternative_count) {
                        std::cmp::Ordering::Less => {
                            best_alternative_count = tuple_changes.len();
                            best_alternatives.clear();
                            best_alternatives.push((positive_index, tuple_changes));
                        }
                        std::cmp::Ordering::Equal => {
                            best_alternatives.push((positive_index, tuple_changes));
                        }
                        std::cmp::Ordering::Greater => {}
                    }
                }
            }
        }

        if unchanged {
            return Ok(Some(InnerIndexedProtocolComplementPlan::Unchanged));
        }
        if !best_alternatives.is_empty() {
            let mut combined = vec![Vec::new()];
            for (positive_index, tuple_changes) in best_alternatives {
                if combined.len().saturating_mul(tuple_changes.len()) > max_alternatives {
                    return Err(());
                }
                combined = combined
                    .into_iter()
                    .flat_map(|alternative| {
                        tuple_changes.iter().map(move |(element_index, remaining)| {
                            let mut expanded = alternative.clone();
                            expanded.push((positive_index, *element_index, *remaining));
                            expanded
                        })
                    })
                    .collect();
            }
            return Ok(Some(InnerIndexedProtocolComplementPlan::Alternatives {
                tuple_changes: combined,
            }));
        }
        if exceeded_limit {
            return Err(());
        }
        Ok(None)
    }

    /// Plan all finite indexed protocol complements against the original positive tuple types.
    ///
    /// Combining the sparse alternatives before materialization makes the result independent of
    /// the order in which negative protocols were added.
    fn plan_indexed_protocol_complements(
        &self,
        db: &'db dyn Db,
        protocols: &FxOrderSet<Type<'db>>,
        max_alternatives: usize,
    ) -> Result<Option<InnerIndexedProtocolComplementsPlan<'db>>, ()> {
        let mut handled_protocols = FxOrderSet::default();
        let mut plans = Vec::new();

        for protocol in protocols {
            if !self.negative.contains(protocol) {
                continue;
            }
            let Type::ProtocolInstance(protocol_instance) = protocol else {
                continue;
            };
            let Some(indexed) = protocol_instance.finite_indexed_constraint(db) else {
                continue;
            };
            let Some(plan) =
                self.plan_indexed_protocol_complement(db, &indexed, max_alternatives)?
            else {
                continue;
            };
            handled_protocols.insert(*protocol);

            match plan {
                InnerIndexedProtocolComplementPlan::Eliminated => {
                    return Ok(Some(InnerIndexedProtocolComplementsPlan {
                        handled_protocols,
                        alternatives: Vec::new(),
                    }));
                }
                InnerIndexedProtocolComplementPlan::Unchanged => {}
                InnerIndexedProtocolComplementPlan::Alternatives { tuple_changes } => {
                    plans.push(tuple_changes);
                }
            }
        }

        if handled_protocols.is_empty() {
            return Ok(None);
        }

        let mut alternatives = vec![Vec::new()];
        for tuple_changes in plans {
            if alternatives.len().saturating_mul(tuple_changes.len()) > max_alternatives {
                return Err(());
            }
            alternatives = alternatives
                .into_iter()
                .flat_map(|alternative| {
                    tuple_changes.iter().map(move |tuple_changes| {
                        let mut expanded = alternative.clone();
                        expanded.extend(tuple_changes.iter().copied());
                        expanded
                    })
                })
                .collect();
        }

        let alternatives = alternatives
            .into_iter()
            .filter_map(|changes| self.plan_tuple_replacements(db, changes))
            .collect();
        Ok(Some(InnerIndexedProtocolComplementsPlan {
            handled_protocols,
            alternatives,
        }))
    }

    fn plan_tuple_replacements(
        &self,
        db: &'db dyn Db,
        changes: Vec<(usize, usize, Type<'db>)>,
    ) -> Option<Vec<TupleReplacementPlan<'db>>> {
        let mut replacements = FxOrderMap::<usize, Vec<Type<'db>>>::default();
        for (positive_index, element_index, remaining_element) in changes {
            let elements = replacements.entry(positive_index).or_insert_with(|| {
                let Type::NominalInstance(instance) = self.positive[positive_index] else {
                    return Vec::new();
                };
                instance
                    .own_tuple_spec(db)
                    .map_or_else(Vec::new, |tuple| tuple.all_elements().to_vec())
            });
            let element = *elements.get(element_index)?;
            let refined =
                build_indexed_protocol_planning_intersection(db, [element, remaining_element], []);
            if refined.is_never() {
                return None;
            }
            elements[element_index] = refined;
        }

        let mut replacements = replacements
            .into_iter()
            .map(|(positive_index, elements)| TupleReplacementPlan {
                positive_index,
                elements,
            })
            .collect::<Vec<_>>();
        replacements.sort_unstable_by_key(|replacement| replacement.positive_index);
        Some(replacements)
    }

    fn materialize_indexed_protocol_complements(
        self,
        db: &'db dyn Db,
        plan: InnerIndexedProtocolComplementsPlan<'db>,
    ) -> Vec<Self> {
        let mut alternatives = Vec::with_capacity(plan.alternatives.len());
        for replacements in plan.alternatives {
            let mut alternative = self.clone();
            for protocol in &plan.handled_protocols {
                alternative.negative.swap_remove(protocol);
            }
            for replacement in replacements.iter().rev() {
                alternative
                    .positive
                    .swap_remove_index(replacement.positive_index);
            }
            for replacement in replacements {
                alternative.add_positive(db, Type::heterogeneous_tuple(db, replacement.elements));
            }
            if !alternative.positive.contains(&Type::Never) {
                alternatives.push(alternative);
            }
        }
        alternatives
    }

    /// Return `true` when an intersection excludes every member of an enum class.
    ///
    /// This recognizes enum complements that have become empty, such as
    /// `Color & ~Literal[Color.RED] & ~Literal[Color.BLUE]` for a two-member enum.
    ///
    /// ```python
    /// from enum import Enum
    ///
    /// class Color(Enum):
    ///     RED = 1
    ///     BLUE = 2
    ///
    /// def f(color: Color):
    ///     if color is not Color.RED and color is not Color.BLUE:
    ///         reveal_type(color)  # Never
    /// ```
    fn has_empty_enum_complement(&self, db: &'db dyn Db) -> bool {
        for positive in &self.positive {
            let Type::NominalInstance(instance) = positive else {
                continue;
            };

            let enum_class = instance.class_literal(db);
            let Some(metadata) = enum_metadata(db, enum_class) else {
                continue;
            };

            let mut excluded_names = FxHashSet::default();
            for negative in &self.negative {
                let Some(enum_literal) = negative.as_enum_literal() else {
                    continue;
                };
                if enum_literal.enum_class(db) != enum_class {
                    continue;
                }

                let name = enum_literal.name(db);
                let canonical_name = metadata.resolve_member(name).unwrap_or(name);
                excluded_names.insert(canonical_name.clone());
            }

            if excluded_names.is_empty() {
                continue;
            }

            if metadata
                .members
                .keys()
                .all(|name| excluded_names.contains(name))
            {
                return true;
            }
        }

        false
    }

    /// Adds a positive type to this intersection.
    fn add_positive(&mut self, db: &'db dyn Db, mut new_positive: Type<'db>) {
        // `Never & T` -> `Never`
        if self.positive.contains(&Type::Never) {
            return;
        }

        // `T & Never` -> `Never`
        if new_positive.is_never() {
            *self = Self::default();
            self.positive.insert(Type::Never);
            return;
        }

        // `T & Divergent` -> `Divergent`. Conceptually, `Divergent` behaves like `Never` here and
        // dominates intersections. However, `Divergent` is actually a dynamic/gradual type, so
        // `~Divergent` acts like `Divergent` rather than dropping out like `~Never` does.
        // `Divergent` also gets a lot of special handling in cycle recovery.
        if new_positive.is_divergent() {
            *self = Self::default();
            self.positive.insert(new_positive);
            return;
        }
        // `Divergent & T` -> `Divergent`
        if self.positive.iter().any(Type::is_divergent) {
            return;
        }

        // A runtime class value of `TypeForm[T]` has type `type[T]`.
        match new_positive {
            Type::TypeForm(typeform) => {
                if let Some(narrowed) = SubclassOfType::try_from_instance(
                    db,
                    typeform.type_argument(db).resolve_type_alias(db),
                ) && self.positive.swap_remove(&KnownClass::Type.to_instance(db))
                {
                    new_positive = narrowed;
                }
            }
            Type::NominalInstance(instance) if instance.has_known_class(db, KnownClass::Type) => {
                if let Some((index, narrowed)) =
                    self.positive
                        .iter()
                        .enumerate()
                        .find_map(|(index, positive)| match positive {
                            Type::TypeForm(typeform) => SubclassOfType::try_from_instance(
                                db,
                                typeform.type_argument(db).resolve_type_alias(db),
                            )
                            .map(|narrowed| (index, narrowed)),
                            _ => None,
                        })
                {
                    self.positive.swap_remove_index(index);
                    new_positive = narrowed;
                }
            }
            _ => {}
        }

        match new_positive {
            // `LiteralString & AlwaysTruthy` -> `LiteralString & ~Literal[""]`
            Type::AlwaysTruthy if self.positive.contains(&Type::literal_string()) => {
                self.add_negative(db, Type::string_literal(db, ""));
            }
            // `LiteralString & AlwaysFalsy` -> `Literal[""]`
            Type::AlwaysFalsy if self.positive.swap_remove(&Type::literal_string()) => {
                self.add_positive(db, Type::string_literal(db, ""));
            }
            // `AlwaysTruthy & LiteralString` -> `LiteralString & ~Literal[""]`
            Type::LiteralValue(literal)
                if literal.is_literal_string()
                    && self.positive.swap_remove(&Type::AlwaysTruthy) =>
            {
                self.add_positive(db, Type::literal_string());
                self.add_negative(db, Type::string_literal(db, ""));
            }
            // `AlwaysFalsy & LiteralString` -> `Literal[""]`
            Type::LiteralValue(literal)
                if literal.is_literal_string() && self.positive.swap_remove(&Type::AlwaysFalsy) =>
            {
                self.add_positive(db, Type::string_literal(db, ""));
            }
            // `LiteralString & ~AlwaysTruthy` -> `LiteralString & AlwaysFalsy` -> `Literal[""]`
            Type::LiteralValue(literal)
                if literal.is_literal_string()
                    && self.negative.swap_remove(&Type::AlwaysTruthy) =>
            {
                self.add_positive(db, Type::string_literal(db, ""));
            }
            // `LiteralString & ~AlwaysFalsy` -> `LiteralString & ~Literal[""]`
            Type::LiteralValue(literal)
                if literal.is_literal_string() && self.negative.swap_remove(&Type::AlwaysFalsy) =>
            {
                self.add_positive(db, Type::literal_string());
                self.add_negative(db, Type::string_literal(db, ""));
            }

            _ => {
                let positive_as_instance = new_positive.as_nominal_instance();

                if let Some(instance) = positive_as_instance
                    && instance.is_object()
                {
                    // `object & T` -> `T`; it is always redundant to add `object` to an intersection
                    return;
                }

                let addition_is_bool_instance = positive_as_instance
                    .is_some_and(|instance| instance.has_known_class(db, KnownClass::Bool));

                if let Some((index, refined)) =
                    self.positive
                        .iter()
                        .enumerate()
                        .find_map(|(index, existing_positive)| {
                            refine_tuple_protocol_intersection(db, *existing_positive, new_positive)
                                .map(|refined| (index, refined))
                        })
                {
                    self.positive.swap_remove_index(index);
                    self.add_positive(db, refined);
                    return;
                }

                for (index, existing_positive) in self.positive.iter().enumerate() {
                    match existing_positive {
                        // `AlwaysTruthy & bool` -> `Literal[True]`
                        Type::AlwaysTruthy if addition_is_bool_instance => {
                            new_positive = Type::bool_literal(true);
                        }
                        // `AlwaysFalsy & bool` -> `Literal[False]`
                        Type::AlwaysFalsy if addition_is_bool_instance => {
                            new_positive = Type::bool_literal(false);
                        }
                        Type::NominalInstance(instance)
                            if instance.has_known_class(db, KnownClass::Bool) =>
                        {
                            match new_positive {
                                // `bool & AlwaysTruthy` -> `Literal[True]`
                                Type::AlwaysTruthy => {
                                    new_positive = Type::bool_literal(true);
                                }
                                // `bool & AlwaysFalsy` -> `Literal[False]`
                                Type::AlwaysFalsy => {
                                    new_positive = Type::bool_literal(false);
                                }
                                _ => continue,
                            }
                        }
                        _ => continue,
                    }
                    self.positive.swap_remove_index(index);
                    break;
                }

                if addition_is_bool_instance {
                    for (index, existing_negative) in self.negative.iter().enumerate() {
                        match existing_negative {
                            // `bool & ~Literal[False]` -> `Literal[True]`
                            // `bool & ~Literal[True]` -> `Literal[False]`
                            Type::LiteralValue(literal) => match literal.kind() {
                                LiteralValueTypeKind::Bool(bool_value) => {
                                    new_positive = Type::bool_literal(!bool_value);
                                }
                                _ => continue,
                            },
                            // `bool & ~AlwaysTruthy` -> `Literal[False]`
                            Type::AlwaysTruthy => {
                                new_positive = Type::bool_literal(false);
                            }
                            // `bool & ~AlwaysFalsy` -> `Literal[True]`
                            Type::AlwaysFalsy => {
                                new_positive = Type::bool_literal(true);
                            }
                            _ => continue,
                        }
                        self.negative.swap_remove_index(index);
                        break;
                    }
                }

                let mut to_remove = SmallVec::<[usize; 1]>::new();
                for (index, existing_positive) in self.positive.iter().enumerate() {
                    // S & T = S    if S <: T
                    if existing_positive.is_redundant_with(db, new_positive) {
                        return;
                    }
                    // same rule, reverse order
                    if new_positive.is_redundant_with(db, *existing_positive) {
                        to_remove.push(index);
                    }
                    // A & B = Never    if A and B are disjoint
                    if new_positive.is_disjoint_from(db, *existing_positive) {
                        *self = Self::default();
                        self.positive.insert(Type::Never);
                        return;
                    }
                }
                for index in to_remove.into_iter().rev() {
                    self.positive.swap_remove_index(index);
                }

                let mut to_remove = SmallVec::<[usize; 1]>::new();
                for (index, existing_negative) in self.negative.iter().enumerate() {
                    // S & ~T = Never    if S <: T
                    if new_positive.is_subtype_of(db, *existing_negative) {
                        *self = Self::default();
                        self.positive.insert(Type::Never);
                        return;
                    }
                    // A & ~B = A    if A and B are disjoint
                    if existing_negative.is_disjoint_from(db, new_positive) {
                        to_remove.push(index);
                    }
                }
                for index in to_remove.into_iter().rev() {
                    self.negative.swap_remove_index(index);
                }

                self.positive.insert(new_positive);
            }
        }
    }

    /// Adds a negative type to this intersection.
    fn add_negative(&mut self, db: &'db dyn Db, new_negative: Type<'db>) {
        // `Never & ~T` -> `Never`.
        if self.positive.contains(&Type::Never) {
            return;
        }

        // `Divergent & ~T` -> `Divergent`.
        if self.positive.iter().any(Type::is_divergent) {
            debug_assert_eq!(self.positive.len(), 1, "`Divergent` should be alone");
            return;
        }

        if let Some(negated_divergent) = new_negative.negated_divergent() {
            *self = Self::default();
            self.positive.insert(negated_divergent);
            return;
        }

        let contains_bool = || {
            self.positive
                .iter()
                .filter_map(|ty| ty.as_nominal_instance())
                .filter_map(|instance| instance.known_class(db))
                .any(KnownClass::is_bool)
        };

        match new_negative {
            Type::Intersection(inter) => {
                for pos in inter.positive(db) {
                    self.add_negative(db, *pos);
                }
                for neg in inter.negative(db) {
                    self.add_positive(db, *neg);
                }
            }
            Type::Never => {
                // Adding ~Never to an intersection is a no-op.
            }
            Type::NominalInstance(instance) if instance.is_object() => {
                // Adding ~object to an intersection results in Never.
                *self = Self::default();
                self.positive.insert(Type::Never);
            }
            ty @ Type::Dynamic(_) => {
                // Adding any of these types to the negative side of an intersection
                // is equivalent to adding it to the positive side. We do this to
                // simplify the representation.
                self.add_positive(db, ty);
            }
            // `bool & ~AlwaysTruthy` -> `bool & Literal[False]`
            Type::AlwaysTruthy if contains_bool() => {
                self.add_positive(db, Type::bool_literal(false));
            }
            // `bool & ~Literal[True]` -> `bool & Literal[False]`
            Type::LiteralValue(literal) if literal.as_bool() == Some(true) && contains_bool() => {
                self.add_positive(db, Type::bool_literal(false));
            }
            // `LiteralString & ~AlwaysTruthy` -> `LiteralString & Literal[""]`
            Type::AlwaysTruthy if self.positive.contains(&Type::literal_string()) => {
                self.add_positive(db, Type::string_literal(db, ""));
            }
            // `bool & ~AlwaysFalsy` -> `bool & Literal[True]`
            Type::AlwaysFalsy if contains_bool() => {
                self.add_positive(db, Type::bool_literal(true));
            }
            // `bool & ~Literal[False]` -> `bool & Literal[True]`
            Type::LiteralValue(literal) if literal.as_bool() == Some(false) && contains_bool() => {
                self.add_positive(db, Type::bool_literal(true));
            }
            // `LiteralString & ~AlwaysFalsy` -> `LiteralString & ~Literal[""]`
            Type::AlwaysFalsy if self.positive.contains(&Type::literal_string()) => {
                self.add_negative(db, Type::string_literal(db, ""));
            }
            _ => {
                let new_negative_enum = new_negative.as_enum_literal();
                let mut to_remove = SmallVec::<[usize; 1]>::new();
                for (index, existing_negative) in self.negative.iter().enumerate() {
                    if let Some(new_enum) = new_negative_enum
                        && existing_negative
                            .as_enum_literal()
                            .is_some_and(|existing_enum| {
                                existing_enum.enum_class(db) == new_enum.enum_class(db)
                            })
                    {
                        if existing_negative.as_enum_literal() == Some(new_enum) {
                            return;
                        }
                        continue;
                    }

                    // ~S & ~T = ~T    if S <: T
                    if existing_negative.is_redundant_with(db, new_negative) {
                        to_remove.push(index);
                    }
                    // same rule, reverse order
                    if new_negative.is_subtype_of(db, *existing_negative) {
                        return;
                    }
                }
                for index in to_remove.into_iter().rev() {
                    self.negative.swap_remove_index(index);
                }

                for existing_positive in &self.positive {
                    if let Some(new_enum) = new_negative_enum {
                        if let Some(existing_enum) = existing_positive.as_enum_literal()
                            && existing_enum.enum_class(db) == new_enum.enum_class(db)
                        {
                            if existing_enum == new_enum {
                                *self = Self::default();
                                self.positive.insert(Type::Never);
                            }
                            return;
                        }

                        if existing_positive
                            .as_nominal_instance()
                            .is_some_and(|instance| {
                                instance.class_literal(db) == new_enum.enum_class(db)
                            })
                        {
                            continue;
                        }
                    }

                    // S & ~T = Never    if S <: T
                    if existing_positive.is_subtype_of(db, new_negative) {
                        *self = Self::default();
                        self.positive.insert(Type::Never);
                        return;
                    }
                    // A & ~B = A    if A and B are disjoint
                    if existing_positive.is_disjoint_from(db, new_negative) {
                        return;
                    }
                }

                self.negative.insert(new_negative);
            }
        }
    }

    /// Tries to simplify any constrained typevars in the intersection.
    ///
    /// We must preserve the constrained `TypeVar` itself in the result, even if only a single
    /// compatible constraint remains, because other occurrences of the same `TypeVar` still need
    /// to correlate with it (for example, when returning a narrowed value as `T`).
    ///
    /// - If the intersection contains negative entries for all but one of the constraints, we can
    ///   add that remaining constraint as a positive entry.
    ///
    /// - If the intersection contains negative entries for all of the constraints, the overall
    ///   intersection is `Never`.
    fn simplify_constrained_typevars(&mut self, db: &'db dyn Db) {
        let mut to_add = SmallVec::<[Type<'db>; 1]>::new();

        for ty in &self.positive {
            let Type::TypeVar(bound_typevar) = ty else {
                continue;
            };
            let Some(TypeVarBoundOrConstraints::Constraints(constraints)) =
                bound_typevar.typevar(db).bound_or_constraints(db)
            else {
                continue;
            };

            // Determine which constraints appear as negative entries in the intersection.
            let constraints = constraints.elements(db);
            let mut remaining_constraints: Vec<_> = constraints.iter().copied().map(Some).collect();
            for negative in &self.negative {
                // This linear search should be fine as long as we don't encounter typevars with
                // thousands of constraints.
                let matching_constraints = constraints
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.is_subtype_of(db, *negative));
                for (constraint_index, _) in matching_constraints {
                    remaining_constraints[constraint_index] = None;
                }
            }

            let mut iter = remaining_constraints.into_iter().flatten();
            let Some(remaining_constraint) = iter.next() else {
                // All of the typevar constraints have been removed, so the entire intersection is
                // `Never`.
                *self = Self::default();
                self.positive.insert(Type::Never);
                return;
            };

            let more_than_one_remaining_constraint = iter.next().is_some();
            if more_than_one_remaining_constraint {
                // This typevar cannot be simplified.
                continue;
            }

            // Only one typevar constraint remains. Adding it as a positive element lets the normal
            // intersection simplification remove any incompatible negatives, while keeping the
            // original typevar in the result.
            to_add.push(remaining_constraint);
        }

        for remaining_constraint in to_add {
            self.add_positive(db, remaining_constraint);
        }
    }

    fn build(mut self, db: &'db dyn Db) -> Type<'db> {
        if self.has_empty_enum_complement(db) {
            return Type::Never;
        }

        self.simplify_constrained_typevars(db);

        // If any typevars are in `self.positive`, speculatively solve all bounded type variables
        // to their upper bound and all constrained type variables to the union of their constraints.
        // If that speculative intersection simplifies to `Never`, this intersection must also simplify
        // to `Never`.
        if self
            .positive
            .iter()
            .any(|ty| matches!(ty, Type::TypeVar(_) | Type::NewTypeInstance(_)))
        {
            let speculative =
                expand_intersection_typevars_and_newtypes(db, &self.positive, &self.negative);
            if speculative.is_never() {
                return Type::Never;
            }
        }

        if let Some(complement) =
            EnumComplement::from_intersection_parts(db, &self.positive, &self.negative)
        {
            return Type::EnumComplement(complement);
        }

        match (self.positive.len(), self.negative.len()) {
            (0, 0) => Type::object(),
            (1, 0) => self.positive[0],
            _ => {
                self.positive.shrink_to_fit();
                self.negative.shrink_to_fit();
                Type::Intersection(IntersectionType::new(db, self.positive, self.negative))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        IntersectionBuilder, MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES,
        MAX_NON_RECURSIVE_UNION_LITERALS, TupleProtocolComplementPlan, Type, UnionBuilder,
        UnionType, plan_indexed_protocol_complement,
    };

    use crate::db::tests::{TestDb, setup_db};
    use crate::place::{global_symbol, known_module_symbol};
    use crate::types::enums::enum_member_literals;
    use crate::types::match_pattern::exact_sequence_pattern_type;
    use crate::types::type_alias::TypeAliasType;
    use crate::types::{KnownClass, KnownInstanceType, Truthiness};

    use ruff_db::system::DbWithWritableSystem as _;
    use ty_module_resolver::KnownModule;

    fn exact_sequence_protocol<'db>(db: &'db TestDb, elements: &[Type<'db>]) -> Type<'db> {
        let Type::Intersection(pattern) = exact_sequence_pattern_type(db, elements) else {
            panic!("Expected exact sequence pattern to be an intersection");
        };
        *pattern
            .positive(db)
            .iter()
            .find(|positive| matches!(positive, Type::ProtocolInstance(_)))
            .expect("Expected exact sequence pattern to contain a protocol")
    }

    fn assert_indexed_protocol_expansion_fits(builder: &IntersectionBuilder<'_>) {
        let protocols = builder.indexed_protocol_negatives();
        assert!(
            builder
                .plan_indexed_protocol_complements(&protocols)
                .is_ok()
        );
    }

    #[test]
    fn build_union_no_elements() {
        let db = setup_db();

        let empty_union = UnionBuilder::new(&db).build();
        assert_eq!(empty_union, Type::Never);
    }

    #[test]
    fn build_union_single_element() {
        let db = setup_db();

        let t0 = Type::int_literal(0);
        let union = UnionType::from_elements(&db, [t0]);
        assert_eq!(union, t0);
    }

    #[test]
    fn build_union_two_elements() {
        let db = setup_db();

        let t0 = Type::int_literal(0);
        let t1 = Type::int_literal(1);
        let union = UnionType::from_elements(&db, [t0, t1]).expect_union();

        assert_eq!(union.elements(&db), &[t0, t1]);
    }

    #[test]
    fn tuple_protocol_complement_plan_refines_each_element() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let element = UnionType::from_two_elements(&db, int, str);
        let tuple = Type::heterogeneous_tuple(&db, [element, element]);

        assert_eq!(
            plan_indexed_protocol_complement(&db, tuple, &[int, str], usize::MAX),
            TupleProtocolComplementPlan::Alternatives(vec![(0, str), (1, int)])
        );
    }

    #[test]
    fn tuple_protocol_complement_expansion_is_bounded() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let element = UnionType::from_two_elements(&db, int, str);
        let length = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES + 1;
        let tuple = Type::heterogeneous_tuple(&db, std::iter::repeat_n(element, length));
        assert_eq!(
            plan_indexed_protocol_complement(
                &db,
                tuple,
                &vec![int; length],
                MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES,
            ),
            TupleProtocolComplementPlan::ExceedsLimit
        );
        let protocol = exact_sequence_protocol(&db, &vec![int; length]);

        let result = IntersectionBuilder::new(&db)
            .add_positive(tuple)
            .add_negative(protocol)
            .build();
        let Type::Intersection(result) = result else {
            panic!("Expected complement beyond the limit to remain symbolic");
        };
        assert!(result.negative(&db).contains(&protocol));
    }

    #[test]
    fn tuple_protocol_complement_budget_is_shared_across_positive_union_arms() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let bytes = KnownClass::Bytes.to_instance(&db);
        let length = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES / 2 + 1;
        let tuple_with_str = Type::heterogeneous_tuple(
            &db,
            std::iter::repeat_n(UnionType::from_two_elements(&db, int, str), length),
        );
        let tuple_with_bytes = Type::heterogeneous_tuple(
            &db,
            std::iter::repeat_n(UnionType::from_two_elements(&db, int, bytes), length),
        );
        let tuples = UnionType::from_two_elements(&db, tuple_with_str, tuple_with_bytes);
        let protocol = exact_sequence_protocol(&db, &vec![int; length]);

        let negative_then_positive = IntersectionBuilder::new(&db)
            .add_negative(protocol)
            .add_positive(tuples)
            .build();
        let positive_then_negative = IntersectionBuilder::new(&db)
            .add_positive(tuples)
            .add_negative(protocol)
            .build();

        assert_eq!(negative_then_positive, positive_then_negative);
    }

    #[test]
    fn tuple_protocol_complement_expansion_is_independent_of_union_order() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let bytes = KnownClass::Bytes.to_instance(&db);
        let length = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES / 2 + 1;
        let tuple_with_str = Type::heterogeneous_tuple(
            &db,
            std::iter::repeat_n(UnionType::from_two_elements(&db, int, str), length),
        );
        let tuple_with_bytes = Type::heterogeneous_tuple(
            &db,
            std::iter::repeat_n(UnionType::from_two_elements(&db, int, bytes), length),
        );
        let protocol = exact_sequence_protocol(&db, &vec![int; length]);

        let forward = IntersectionBuilder::new(&db)
            .add_positive(UnionType::from_two_elements(
                &db,
                tuple_with_str,
                tuple_with_bytes,
            ))
            .add_negative(protocol)
            .build();
        let reverse = IntersectionBuilder::new(&db)
            .add_positive(UnionType::from_two_elements(
                &db,
                tuple_with_bytes,
                tuple_with_str,
            ))
            .add_negative(protocol)
            .build();

        for result in [forward, reverse] {
            let Type::Union(result) = result else {
                panic!("Expected the over-budget complement to remain a union");
            };
            assert_eq!(result.elements(&db).len(), 2);
            assert!(result.elements(&db).iter().all(|element| {
                let Type::Intersection(intersection) = element else {
                    return false;
                };
                intersection.negative(&db).contains(&protocol)
            }));
        }
    }

    #[test]
    fn tuple_protocol_complement_expansion_is_independent_of_negative_order() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let bytes = KnownClass::Bytes.to_instance(&db);
        let length = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES / 2 + 1;
        let element = UnionType::from_elements(&db, [int, str, bytes]);
        let tuple = Type::heterogeneous_tuple(&db, std::iter::repeat_n(element, length));
        let int_protocol = exact_sequence_protocol(&db, &vec![int; length]);
        let str_protocol = exact_sequence_protocol(&db, &vec![str; length]);

        let int_then_str = IntersectionBuilder::new(&db)
            .add_positive(tuple)
            .add_negative(int_protocol)
            .add_negative(str_protocol)
            .build();
        let str_then_int = IntersectionBuilder::new(&db)
            .add_positive(tuple)
            .add_negative(str_protocol)
            .add_negative(int_protocol)
            .build();

        for result in [int_then_str, str_then_int] {
            let Type::Intersection(result) = result else {
                panic!("Expected over-budget complements to remain symbolic");
            };
            assert!(result.negative(&db).contains(&int_protocol));
            assert!(result.negative(&db).contains(&str_protocol));
        }
    }

    #[test]
    fn asymmetric_tuple_protocol_complements_are_independent_of_negative_order() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let element = UnionType::from_two_elements(&db, int, str);
        let length = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES + 1;
        let tuple = Type::heterogeneous_tuple(&db, std::iter::repeat_n(element, length));
        let narrow_protocol = exact_sequence_protocol(
            &db,
            &std::iter::once(str)
                .chain(std::iter::repeat_n(Type::object(), length - 1))
                .collect::<Vec<_>>(),
        );
        let wide_protocol = exact_sequence_protocol(&db, &vec![int; length]);

        let narrow_then_wide = IntersectionBuilder::new(&db)
            .add_positive(tuple)
            .add_negative(narrow_protocol)
            .add_negative(wide_protocol)
            .build();
        let wide_then_narrow = IntersectionBuilder::new(&db)
            .add_positive(tuple)
            .add_negative(wide_protocol)
            .add_negative(narrow_protocol)
            .build();

        for result in [narrow_then_wide, wide_then_narrow] {
            let Type::Intersection(result) = result else {
                panic!("Expected over-budget complements to remain symbolic");
            };
            assert!(result.negative(&db).contains(&narrow_protocol));
            assert!(result.negative(&db).contains(&wide_protocol));
        }
    }

    #[test]
    fn eliminated_tuple_arm_returns_complement_expansion_budget() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let element = UnionType::from_two_elements(&db, int, str);
        let wide_length = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES;
        let singleton = Type::heterogeneous_tuple(&db, [int]);
        let wide = Type::heterogeneous_tuple(&db, std::iter::repeat_n(element, wide_length));
        let tuples = UnionType::from_two_elements(&db, singleton, wide);
        let singleton_protocol = exact_sequence_protocol(&db, &[int]);
        let wide_protocol = exact_sequence_protocol(&db, &vec![int; wide_length]);

        let builder = IntersectionBuilder::new(&db)
            .add_positive(tuples)
            .add_negative(singleton_protocol)
            .add_negative(wide_protocol);
        assert_indexed_protocol_expansion_fits(&builder);
        let result = builder.build();

        let Type::Union(result) = result else {
            panic!("Expected the wide tuple complement to expand, got {result:?}");
        };
        assert_eq!(
            result.elements(&db).len(),
            MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES
        );
    }

    #[test]
    fn tuple_protocol_complement_plan_is_bounded() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let bytes = KnownClass::Bytes.to_instance(&db);
        let zero = Type::int_literal(0);
        let length = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES;
        let first_tuple = Type::heterogeneous_tuple(
            &db,
            std::iter::once(UnionType::from_two_elements(&db, int, str))
                .chain(std::iter::repeat_n(int, length - 1)),
        );
        let second_tuple = Type::heterogeneous_tuple(
            &db,
            std::iter::once(UnionType::from_two_elements(&db, int, bytes)).chain(
                std::iter::repeat_n(UnionType::from_two_elements(&db, int, str), length - 1),
            ),
        );
        let first_protocol = exact_sequence_protocol(
            &db,
            &std::iter::once(str)
                .chain(std::iter::repeat_n(zero, length - 1))
                .collect::<Vec<_>>(),
        );
        let second_protocol = exact_sequence_protocol(&db, &vec![int; length]);

        let builder = IntersectionBuilder::new(&db)
            .add_positive(first_tuple)
            .add_positive(second_tuple)
            .add_negative(first_protocol)
            .add_negative(second_protocol);
        assert_indexed_protocol_expansion_fits(&builder);
    }

    #[test]
    fn tuple_protocol_complement_materialization_budget_is_cumulative() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let bytes = KnownClass::Bytes.to_instance(&db);
        let int_or_str = UnionType::from_two_elements(&db, int, str);
        let trailing = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES;
        let first_tuple = Type::heterogeneous_tuple(
            &db,
            [int_or_str, int_or_str]
                .into_iter()
                .chain(std::iter::repeat_n(int, trailing)),
        );
        let second_tuple = Type::heterogeneous_tuple(
            &db,
            [UnionType::from_two_elements(&db, int, bytes), str]
                .into_iter()
                .chain(std::iter::repeat_n(int_or_str, trailing)),
        );
        let first_protocol = exact_sequence_protocol(
            &db,
            &std::iter::once(str)
                .chain(std::iter::repeat_n(Type::object(), trailing + 1))
                .collect::<Vec<_>>(),
        );
        let second_protocol = exact_sequence_protocol(
            &db,
            &[Type::object(), str]
                .into_iter()
                .chain(std::iter::repeat_n(int, trailing))
                .collect::<Vec<_>>(),
        );

        for [first, second] in [
            [first_protocol, second_protocol],
            [second_protocol, first_protocol],
        ] {
            let builder = IntersectionBuilder::new(&db)
                .add_positive(first_tuple)
                .add_positive(second_tuple)
                .add_negative(first)
                .add_negative(second);
            assert_indexed_protocol_expansion_fits(&builder);
        }
    }

    #[test]
    fn indexed_protocol_planning_keeps_nested_complement_symbolic() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let element = UnionType::from_two_elements(&db, int, str);
        let length = 12;
        let inner_tuple = Type::heterogeneous_tuple(&db, std::iter::repeat_n(element, length));
        let inner_protocol = exact_sequence_protocol(&db, &vec![int; length]);
        let remaining = super::build_indexed_protocol_planning_intersection(
            &db,
            [inner_tuple],
            [inner_protocol],
        );

        let Type::Intersection(remaining) = remaining else {
            panic!("Expected nested tuple complement to remain symbolic");
        };
        assert!(remaining.negative(&db).contains(&inner_protocol));
    }

    #[test]
    fn same_position_nested_tuple_complements_remain_symbolic() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let bytes = KnownClass::Bytes.to_instance(&db);
        let element = UnionType::from_elements(&db, [int, str, bytes]);
        let length = 8;
        let inner_tuple = Type::heterogeneous_tuple(&db, std::iter::repeat_n(element, length));
        let int_protocol = exact_sequence_protocol(&db, &vec![int; length]);
        let str_protocol = exact_sequence_protocol(&db, &vec![str; length]);
        let int_remaining =
            super::build_indexed_protocol_planning_intersection(&db, [inner_tuple], [int_protocol]);
        let str_remaining =
            super::build_indexed_protocol_planning_intersection(&db, [inner_tuple], [str_protocol]);
        let outer_tuple = Type::heterogeneous_tuple(&db, std::iter::repeat_n(inner_tuple, length));
        let mut inner = super::InnerIntersectionBuilder::default();
        inner.add_positive(&db, outer_tuple);

        let replacements = inner
            .plan_tuple_replacements(&db, vec![(0, 0, int_remaining), (0, 0, str_remaining)])
            .expect("Expected nested tuple replacements to remain possible");
        let [replacement] = replacements.as_slice() else {
            panic!("Expected one outer tuple replacement");
        };
        let Type::Intersection(remaining) = replacement.elements[0] else {
            panic!("Expected merged nested tuple complements to remain symbolic");
        };
        assert!(remaining.negative(&db).contains(&int_protocol));
        assert!(remaining.negative(&db).contains(&str_protocol));
    }

    #[test]
    fn equal_cost_tuple_protocol_complements_are_order_independent() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let bytes = KnownClass::Bytes.to_instance(&db);
        let int_or_str = UnionType::from_two_elements(&db, int, str);
        let trailing = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES + 1;
        let first_tuple = Type::heterogeneous_tuple(
            &db,
            [int_or_str, int_or_str]
                .into_iter()
                .chain(std::iter::repeat_n(int, trailing)),
        );
        let second_tuple = Type::heterogeneous_tuple(
            &db,
            [UnionType::from_two_elements(&db, int, bytes), str]
                .into_iter()
                .chain(std::iter::repeat_n(int_or_str, trailing)),
        );
        let first_protocol = exact_sequence_protocol(
            &db,
            &std::iter::once(str)
                .chain(std::iter::repeat_n(Type::object(), trailing + 1))
                .collect::<Vec<_>>(),
        );
        let second_protocol = exact_sequence_protocol(
            &db,
            &[Type::object(), str]
                .into_iter()
                .chain(std::iter::repeat_n(int, trailing))
                .collect::<Vec<_>>(),
        );
        let build = |first, second| {
            IntersectionBuilder::new(&db)
                .add_positive(first_tuple)
                .add_positive(second_tuple)
                .add_negative(first)
                .add_negative(second)
                .build()
        };

        let first_then_second = build(first_protocol, second_protocol);
        let second_then_first = build(second_protocol, first_protocol);

        assert!(first_then_second.is_never());
        assert_eq!(first_then_second, second_then_first);
    }

    #[test]
    fn eliminating_tuple_protocol_precedes_cartesian_limit() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let element = UnionType::from_two_elements(&db, int, str);
        let length = 12;
        let tuple = Type::heterogeneous_tuple(&db, std::iter::repeat_n(element, length));
        let int_protocol = exact_sequence_protocol(&db, &vec![int; length]);
        let str_protocol = exact_sequence_protocol(&db, &vec![str; length]);
        let object_protocol = exact_sequence_protocol(&db, &vec![Type::object(); length]);

        for protocols in [
            [int_protocol, str_protocol, object_protocol],
            [int_protocol, object_protocol, str_protocol],
            [str_protocol, int_protocol, object_protocol],
            [str_protocol, object_protocol, int_protocol],
            [object_protocol, int_protocol, str_protocol],
            [object_protocol, str_protocol, int_protocol],
        ] {
            let result = IntersectionBuilder::new(&db)
                .add_positive(tuple)
                .add_negative(protocols[0])
                .add_negative(protocols[1])
                .add_negative(protocols[2])
                .build();
            assert!(result.is_never());
        }
    }

    #[test]
    fn tuple_protocol_complement_considers_all_positive_tuples() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let int_or_str = UnionType::from_two_elements(&db, int, str);
        let length = MAX_INDEXED_PROTOCOL_COMPLEMENT_ALTERNATIVES + 1;
        let expansive = Type::heterogeneous_tuple(&db, std::iter::repeat_n(int_or_str, length));
        let disjoint = Type::heterogeneous_tuple(
            &db,
            [str, Type::object()]
                .into_iter()
                .chain(std::iter::repeat_n(int_or_str, length - 2)),
        );
        let protocol = exact_sequence_protocol(&db, &vec![int; length]);
        let build = |first, second, negative| {
            let mut builder = IntersectionBuilder::new(&db)
                .add_positive(first)
                .add_positive(second);
            if negative {
                builder = builder.add_negative(protocol);
            }
            builder.build()
        };

        let expansive_then_disjoint = build(expansive, disjoint, true);
        let disjoint_then_expansive = build(disjoint, expansive, true);

        assert!(expansive_then_disjoint.is_equivalent_to(&db, build(expansive, disjoint, false)));
        assert!(disjoint_then_expansive.is_equivalent_to(&db, build(disjoint, expansive, false)));
        for result in [expansive_then_disjoint, disjoint_then_expansive] {
            let Type::Intersection(result) = result else {
                panic!("Expected both tuple constraints to remain in an intersection");
            };
            assert!(!result.negative(&db).contains(&protocol));
        }
    }

    #[test]
    fn equal_cost_tuple_complement_bases_preserve_same_precision() {
        let db = setup_db();
        let int = KnownClass::Int.to_instance(&db);
        let str = KnownClass::Str.to_instance(&db);
        let bytes = KnownClass::Bytes.to_instance(&db);
        let int_or_str = UnionType::from_two_elements(&db, int, str);
        let first_tuple = Type::heterogeneous_tuple(&db, [int_or_str, int_or_str]);
        let second_tuple = Type::heterogeneous_tuple(
            &db,
            [UnionType::from_two_elements(&db, int, bytes), int_or_str],
        );
        let expected = Type::heterogeneous_tuple(&db, [int_or_str, str]);
        let protocol = exact_sequence_protocol(&db, &[int, int]);

        for [first, second] in [[first_tuple, second_tuple], [second_tuple, first_tuple]] {
            let result = IntersectionBuilder::new(&db)
                .add_positive(first)
                .add_positive(second)
                .add_negative(protocol)
                .build();
            assert!(result.is_assignable_to(&db, expected));
        }
    }

    fn map_marker<'db>(ty: &Type<'db>, marker: Type<'db>, replacement: Type<'db>) -> Type<'db> {
        if *ty == marker { replacement } else { *ty }
    }

    #[test]
    fn map_rebuilds_prefix_for_literal_widening() {
        let db = setup_db();

        let marker = KnownClass::Str.to_instance(&db);
        let literal_limit =
            i64::try_from(MAX_NON_RECURSIVE_UNION_LITERALS).expect("literal limit fits in i64");
        let widening_literal = Type::int_literal(literal_limit);
        let expected = KnownClass::Int.to_instance(&db);

        let elements = (0..literal_limit).map(Type::int_literal).chain([marker]);
        let union = UnionType::from_elements(&db, elements).expect_union();

        assert_eq!(
            union.map(&db, |ty| map_marker(ty, marker, widening_literal)),
            expected
        );
        assert_eq!(
            union.map_leave_aliases(&db, |ty| map_marker(ty, marker, widening_literal)),
            expected
        );
        assert_eq!(
            union.try_map(&db, |ty| Some(map_marker(ty, marker, widening_literal))),
            Some(expected)
        );
    }

    #[test]
    fn map_preserves_alias_unpacking_behavior() {
        let mut db = setup_db();
        db.write_dedented("/src/a.py", "type Alias = int").unwrap();

        let module = ruff_db::files::system_path_to_file(&db, "/src/a.py").unwrap();
        let alias_ty = global_symbol(&db, module, "Alias").place.expect_type();
        let Type::KnownInstance(KnownInstanceType::TypeAliasType(TypeAliasType::PEP695(alias))) =
            alias_ty
        else {
            panic!("Expected `Alias` to be a type alias");
        };

        let alias = Type::TypeAlias(TypeAliasType::PEP695(alias));
        let str_instance = KnownClass::Str.to_instance(&db);
        let union_ty = UnionType::from_elements_leave_aliases(&db, [alias, str_instance]);
        let union = union_ty.expect_union();
        let unpacked =
            UnionType::from_elements(&db, [KnownClass::Int.to_instance(&db), str_instance]);

        assert_eq!(union.map(&db, |ty| *ty), unpacked);
        assert_eq!(union.try_map(&db, |ty| Some(*ty)), Some(unpacked));
        assert_eq!(union.map_leave_aliases(&db, |ty| *ty), union_ty);
    }

    #[test]
    fn build_intersection_empty_intersection_equals_object() {
        let db = setup_db();

        let intersection = IntersectionBuilder::new(&db).build();
        assert_eq!(intersection, Type::object());
    }

    #[test]
    fn ordinary_intersection_has_no_indexed_protocol_negatives() {
        let db = setup_db();
        let builder = IntersectionBuilder::new(&db)
            .add_positive(KnownClass::Int.to_instance(&db))
            .add_negative(Type::int_literal(1));

        assert!(builder.indexed_protocol_negatives().is_empty());
    }

    #[test]
    fn build_intersection_simplify_split_bool() {
        let db = setup_db();

        build_intersection_simplify_split_bool_impl(&db, Type::bool_literal(true));
        build_intersection_simplify_split_bool_impl(&db, Type::bool_literal(false));
        build_intersection_simplify_split_bool_impl(&db, Type::AlwaysTruthy);
        build_intersection_simplify_split_bool_impl(&db, Type::AlwaysFalsy);
    }

    fn build_intersection_simplify_split_bool_impl(db: &TestDb, t_splitter: Type) {
        let bool_value = t_splitter.bool(db) == Truthiness::AlwaysTrue;

        // We add t_object in various orders (in first or second position) in
        // the tests below to ensure that the boolean simplification eliminates
        // everything from the intersection, not just `bool`.
        let t_object = Type::object();
        let t_bool = KnownClass::Bool.to_instance(db);

        let ty = IntersectionBuilder::new(db)
            .add_positive(t_object)
            .add_positive(t_bool)
            .add_negative(t_splitter)
            .build();
        assert_eq!(ty, Type::bool_literal(!bool_value));

        let ty = IntersectionBuilder::new(db)
            .add_positive(t_bool)
            .add_positive(t_object)
            .add_negative(t_splitter)
            .build();
        assert_eq!(ty, Type::bool_literal(!bool_value));

        let ty = IntersectionBuilder::new(db)
            .add_positive(t_object)
            .add_negative(t_splitter)
            .add_positive(t_bool)
            .build();
        assert_eq!(ty, Type::bool_literal(!bool_value));

        let ty = IntersectionBuilder::new(db)
            .add_negative(t_splitter)
            .add_positive(t_object)
            .add_positive(t_bool)
            .build();
        assert_eq!(ty, Type::bool_literal(!bool_value));
    }

    #[test]
    fn build_intersection_enums() {
        let db = setup_db();

        let safe_uuid_class = known_module_symbol(&db, KnownModule::Uuid, "SafeUUID")
            .place
            .ignore_possibly_undefined()
            .unwrap();

        let literals = enum_member_literals(&db, safe_uuid_class.expect_class_literal(), None)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(literals.len(), 3);

        // SafeUUID.safe
        let l_safe = literals[0];
        assert_eq!(l_safe.expect_enum_literal().name(&db), "safe");
        // SafeUUID.unsafe
        let l_unsafe = literals[1];
        assert_eq!(l_unsafe.expect_enum_literal().name(&db), "unsafe");
        // SafeUUID.unknown
        let l_unknown = literals[2];
        assert_eq!(l_unknown.expect_enum_literal().name(&db), "unknown");

        // The enum itself: SafeUUID
        let safe_uuid = l_safe.expect_enum_literal().enum_class_instance(&db);

        {
            let actual = IntersectionBuilder::new(&db)
                .add_positive(safe_uuid)
                .add_negative(l_safe)
                .build();

            assert_eq!(
                actual.display(&db).to_string(),
                "Literal[SafeUUID.unsafe, SafeUUID.unknown]"
            );
        }
        {
            // Same as above, but with the order reversed
            let actual = IntersectionBuilder::new(&db)
                .add_negative(l_safe)
                .add_positive(safe_uuid)
                .build();

            assert_eq!(
                actual.display(&db).to_string(),
                "Literal[SafeUUID.unsafe, SafeUUID.unknown]"
            );
        }
        {
            // Also the same, but now with a nested intersection
            let actual = IntersectionBuilder::new(&db)
                .add_positive(safe_uuid)
                .add_positive(IntersectionBuilder::new(&db).add_negative(l_safe).build())
                .build();

            assert_eq!(
                actual.display(&db).to_string(),
                "Literal[SafeUUID.unsafe, SafeUUID.unknown]"
            );
        }
        {
            let actual = IntersectionBuilder::new(&db)
                .add_negative(l_safe)
                .add_positive(safe_uuid)
                .add_negative(l_unsafe)
                .build();

            assert_eq!(actual.display(&db).to_string(), "Literal[SafeUUID.unknown]");
        }
    }
}
