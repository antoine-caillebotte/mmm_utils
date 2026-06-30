"""
Configuration classes for the MMM model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Interaction helpers
# ---------------------------------------------------------------------------

_TERM_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")


@dataclass(slots=True)
class InteractionFormula:
    """Parsed representation of a single-media interaction formula.

    A formula expresses how one media variable interacts with other variables.
    The syntax is a ``+``-separated list of terms where the literal ``1``
    represents the baseline and any other identifier names a variable that
    modulates the channel.

    Examples
    --------
    ``"1"``         → no interaction (plain channel weight).
    ``"1 + Y2"``    → channel is multiplied by ``(1 + β_{Y:Y2} · Y2)``.
    ``"1 + Y2 + C"`` → channel is multiplied by
                       ``(1 + β_{Y:Y2} · Y2 + β_{Y:C} · C)``.

    Attributes
    ----------
    media_name : str
        The name of the media variable this formula belongs to.
    raw : str
        The original formula string as provided by the user.
    terms : list[str]
        Parsed interaction variable names (i.e., every token that is **not**
        ``"1"``).
    has_baseline : bool
        ``True`` when ``"1"`` is present in the formula.

    Raises
    ------
    ValueError
        If the formula contains unsupported syntax.
    """

    media_name: str
    raw: str
    terms: list[str] = field(init=False)
    has_baseline: bool = field(init=False)

    def __post_init__(self) -> None:
        self.terms, self.has_baseline = self._parse(self.raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(formula: str) -> tuple[list[str], bool]:
        """Parse the formula string into a list of interaction variable names.

        Parameters
        ----------
        formula : str
            Formula string, e.g. ``"1 + Y2 + C"``.

        Returns
        -------
        tuple[list[str], bool]
            A 2-tuple ``(terms, has_baseline)`` where *terms* is the list
            of non-``"1"`` identifier tokens and *has_baseline* is ``True``
            when ``"1"`` appears in the formula.

        Raises
        ------
        ValueError
            If any token is neither ``"1"`` nor a valid Python identifier.
        """
        tokens = [t.strip() for t in formula.split("+")]
        has_baseline = False
        terms: list[str] = []

        for tok in tokens:
            if tok == "1":
                has_baseline = True
            elif tok == "0":
                pass  # explicit "no baseline" marker
            elif _TERM_RE.match(tok):
                terms.append(tok)
            else:
                raise ValueError(
                    f"Invalid token '{tok}' in formula '{formula}'. "
                    "Each term must be '1', '0', or a valid Python identifier."
                )

        return terms, has_baseline

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_default(self) -> bool:
        """Return ``True`` if the formula is the trivial ``"1"`` formula.

        Returns
        -------
        bool
            ``True`` when there are no interaction terms beyond the baseline.
        """
        return self.has_baseline and len(self.terms) == 0


@dataclass
class Interaction:
    """Centralized definition of all media-variable interactions in an MMM.

    Each media channel can optionally have an interaction *formula* that
    describes how its effective coefficient is modulated by other media or
    control variables.  The mathematical model for a channel **Y** with
    formula ``"1 + Z"`` is:

    .. math::

        Y_{\\text{eff}} = \\beta_Y \\cdot (1 + \\beta_{Y:Z} \\cdot Z) \\cdot X_Y

    Parameters
    ----------
    formulas : dict[str, str], optional
        Mapping ``{media_name: formula_string}``.  Absent channels are
        treated as having the default ``"1"`` formula (no interaction).
    is_shared_with : list[tuple[str, ...]] | None, optional
        Each tuple declares that a specific interaction variable is shared
        across a set of media channels.  The **first** element is the
        interaction variable name, and all **remaining** elements are the
        media channels that share a single parameter for that interaction.

        Example: ``("TV", "Digital", "SEA")`` means that the ``TV``
        interaction coefficient is shared between ``Digital`` and ``SEA``.
        The parameter name is auto-generated as
        ``{sorted_medias}:{interact_var}``, e.g. ``Digital,SEA:TV``.
        Other terms in those channels' formulas (e.g. ``Cospirit``) are
        **not** shared and each get their own parameter.
    controls : list[str], optional
        Names of all control variables (every variable that is **not** a
        media channel).  A control not referenced in any formula term is
        treated as having a default interaction coefficient of ``1``
        (i.e. no modulation).  Use :meth:`get_default_controls` to
        retrieve these controls.

    Attributes
    ----------
    formulas : dict[str, str]
        Raw formula strings keyed by media name.
    is_shared_with : list[tuple[str, ...]] | None
        Sharing groups, or ``None`` when no sharing is requested.
    controls : list[str]
        Control variable names.

    Examples
    --------
    >>> ia = Interaction(
    ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3"},
    ...     is_shared_with=[("Y3", "Y1", "Y2")],
    ...     media=["Y1", "Y2"],
    ... )
    >>> ia.get_parameter_name("Y3")
    'beta_interaction_Y3'
    >>> ia.get_lhs_index("Y1", "Y3")   # "Y1,Y2" is index 0 in interaction_Y3
    0
    >>> ia.get_all_interaction_terms()
    {'Y3'}
    """

    formulas: dict[str, str] = field(default_factory=dict)
    is_shared_with: list[tuple[str, ...]] | None = None
    media: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)

    # Derived / cached fields (not part of public API)
    _parsed: dict[str, InteractionFormula] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._parsed = {
            name: InteractionFormula(media_name=name, raw=raw)
            for name, raw in self.formulas.items()
        }
        self._validate_all_terms_defined()
        self._validate_media_has_baseline()
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Run consistency checks on the interaction configuration.

        Raises
        ------
        ValueError
            If a sharing group contains fewer than three members, if the
            interaction variable is empty, or if a media variable listed in
            a sharing group does not have that interaction variable in its
            formula.
        """
        if self.is_shared_with is None:
            return

        for group in self.is_shared_with:
            if len(group) < 3:
                raise ValueError(
                    "Each sharing group must provide one interaction variable "
                    f"and at least two media names, got: {group}"
                )
            interact_var, *media_vars = group
            if not interact_var.strip():
                raise ValueError("Interaction variable name must be a non-empty string")
            for mv in media_vars:
                parsed = self._parsed.get(mv)
                if parsed is None or interact_var not in parsed.terms:
                    raise ValueError(
                        f"Sharing group {group}: media variable '{mv}' does "
                        f"not have '{interact_var}' in its formula "
                        f"(formula: '{self.formulas.get(mv, '1')}')."
                    )

    def _validate_all_terms_defined(self) -> None:
        """Check that every term referenced in any formula is itself defined.

        A term is considered defined if it is a key in ``formulas``, listed
        in ``media``, or listed in ``controls``.

        Raises
        ------
        ValueError
            If any formula term is not defined.
        """
        defined = self.media + self.controls + list(self.formulas.keys())
        for formula in self._parsed.values():
            for term in formula.terms:
                if term not in defined:
                    raise ValueError(
                        f"Formula for '{formula.media_name}' references undefined "
                        f"variable '{term}'. Add it to `formulas`, `media`, or `controls`."
                    )

    def _validate_media_has_baseline(self) -> None:
        """Check that every media channel's formula provides a baseline.

        A media channel has a valid baseline when its formula either contains
        the literal ``1`` (``has_baseline=True``) or references at least one
        other media channel as a term.

        Raises
        ------
        ValueError
            If a media channel has no baseline and no media-channel term.
        """
        if not self.media:
            return
        for media_name in self.media:
            formula = self.parse_formula(media_name)
            if formula.has_baseline:
                continue
            if any(term in self.media for term in formula.terms):
                continue
            raise ValueError(
                f"Media channel '{media_name}' has no baseline. "
                f"Its formula '{formula.raw}' must contain '1' or reference "
                "at least one other media channel."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_formula(self, media_name: str) -> InteractionFormula:
        """Return the parsed ``InteractionFormula`` for *media_name*.

        If *media_name* has no explicit formula, the default ``"1"`` formula
        is returned (and cached).

        Parameters
        ----------
        media_name : str
            Name of the media variable.

        Returns
        -------
        InteractionFormula
            Parsed formula object for the requested channel.
        """
        if media_name not in self._parsed:
            self._parsed[media_name] = InteractionFormula(
                media_name=media_name, raw="1"
            )
        return self._parsed[media_name]

    def get_parameter_name(self, interact_var: str) -> str:
        """Return the PyMC parameter name for a given interaction variable.

        The returned name is always ``"beta_interaction_{interact_var}"``.
        It is independent of media channel; channel-specific indexing is
        handled by :meth:`get_lhs_index`.

        Parameters
        ----------
        interact_var : str
            Interaction variable name.

        Returns
        -------
        str
            PyMC variable name for that interaction variable.

        Examples
        --------
        >>> ia = Interaction(formulas={"Y1": "1 + Y3", "Y3": "1"}, media=["Y1"])
        >>> ia.get_parameter_name("Y3")
        'beta_interaction_Y3'

        >>> ia = Interaction(
        ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        ...     is_shared_with=[("Y3", "Y1", "Y2")],
        ...     media=["Y1", "Y2"],
        ... )
        >>> ia.get_parameter_name("Y3")
        'beta_interaction_Y3'
        """
        return f"beta_interaction_{interact_var}"

    def get_lhs_index(self, media_name: str, interact_var: str) -> int:
        """Return the position of *media_name*'s coefficient in the interaction vector.

        The parameter ``"beta_interaction_{interact_var}"`` is a 1-D PyMC
        variable indexed by the ``"interaction_{interact_var}"`` coordinate.
        This method returns the integer position of *media_name*'s LHS label
        within that coordinate list, which is needed to slice the correct
        scalar value out of the vector prior.

        Parameters
        ----------
        media_name : str
            Media channel whose coefficient index is requested.
        interact_var : str
            Interaction variable name.

        Returns
        -------
        int
            Zero-based index into ``get_coords()["interaction_{interact_var}"]``.

        Examples
        --------
        >>> ia = Interaction(
        ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        ...     is_shared_with=[("Y3", "Y1", "Y2")],
        ...     media=["Y1", "Y2"],
        ... )
        >>> ia.get_lhs_index("Y1", "Y3")  # shared → LHS is "Y1,Y2", index 0
        0
        >>> ia2 = Interaction(
        ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        ...     media=["Y1", "Y2"],
        ... )
        >>> ia2.get_lhs_index("Y2", "Y3")  # unshared → LHS is "Y2", index 1
        1
        """
        lhs = self._get_lhs_label(media_name, interact_var)
        coord_list = self.get_coords()[f"interaction_{interact_var}"]
        return coord_list.index(lhs)

    def resolve_shared_groups(self, media_name: str) -> dict[str, str]:
        """Return the parameter name for every interaction term of *media_name*.

        Parameters
        ----------
        media_name : str
            Name of the media variable.

        Returns
        -------
        dict[str, str]
            Mapping ``{interact_var: parameter_name}`` for every term in
            the formula of *media_name*.

        Examples
        --------
        >>> ia = Interaction(
        ...     formulas={"Y1": "1 + Y3 + C", "Y2": "1 + Y3", "Y3": "1"},
        ...     is_shared_with=[("Y3", "Y1", "Y2")],
        ...     media=["Y1", "Y2"],
        ...     controls=["C"],
        ... )
        >>> ia.resolve_shared_groups("Y1")
        {'Y3': 'beta_interaction_Y3', 'C': 'beta_interaction_C'}
        """
        parsed = self.parse_formula(media_name)
        return {term: self.get_parameter_name(term) for term in parsed.terms}

    def get_all_interaction_terms(self) -> set[str]:
        """Return the set of every distinct interaction variable used in any formula.

        Returns
        -------
        set[str]
            Union of all non-``"1"`` tokens across all registered formulas.

        Examples
        --------
        >>> ia = Interaction(formulas={"Y1": "1 + Y3", "Y2": "1 + C"})
        >>> ia.get_all_interaction_terms() == {"Y3", "C"}
        True
        """
        terms: set[str] = set()
        for parsed in self._parsed.values():
            terms.update(parsed.terms)
        return terms

    def get_default_controls(self) -> list[str]:
        """Return control variables that are not referenced in any formula term.

        Controls absent from all interaction formulas receive the implicit
        default coefficient ``1`` (no modulation).  Callers can use this
        list to inject ``"X:1"`` default entries when building the model.

        Returns
        -------
        list[str]
            Sorted list of control names not mentioned in any formula.

        Examples
        --------
        >>> ia = Interaction(
        ...     formulas={"Y1": "1 + C1", "Y2": "1"},
        ...     controls=["C1", "C2"],
        ... )
        >>> ia.get_default_controls()
        ['C2']
        """
        mentioned = self.get_all_interaction_terms() & self.controls
        return sorted(self.controls - mentioned)

    def get_unique_parameter_names(self) -> set[str]:
        """Return the set of unique PyMC variable names for interaction parameters.

        One vectorized parameter ``"beta_interaction_{var}"`` is produced per
        distinct interaction variable, regardless of how many media channels
        reference it or whether they share a coefficient.

        Returns
        -------
        set[str]
            Set of unique PyMC parameter name strings.

        Examples
        --------
        >>> ia = Interaction(
        ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3"},
        ...     is_shared_with=[("Y3", "Y1", "Y2")],
        ...     media=["Y1", "Y2"],
        ... )
        >>> ia.get_unique_parameter_names()
        {'beta_interaction_Y3'}
        """
        return {f"beta_interaction_{term}" for term in self.get_all_interaction_terms()}

    def get_coords(self) -> dict[str, list[str]]:
        """Return xarray-compatible coordinate lists for every parameter group.

        Keys and their meaning:

        * ``"media"`` — sorted list of all media channel names.
        * ``"controls"`` — sorted list of all control variable names
          (present only when ``controls`` is non-empty).
        * ``"interaction_{var}"`` (one key per distinct interaction variable)
          — the LHS coordinate labels for ``"beta_interaction_{var}"``.
          For a **shared** group the label is the comma-joined sorted names
          of the media in that group (e.g. ``"Digital,SEA"``); for an
          **unshared** channel it is the channel name itself.

        Returns
        -------
        dict[str, list[str]]
            Mapping ``{dimension_name: [coord, ...]}``.

        Examples
        --------
        >>> ia = Interaction(
        ...     formulas={
        ...         "TV": "1",
        ...         "SEA": "1 + TV + Cospirit",
        ...         "Digital": "0 + TV + Cospirit + Concurence",
        ...         "Cospirit": "1",
        ...         "Concurence": "0",
        ...     },
        ...     is_shared_with=[("TV", "Digital", "SEA")],
        ...     media=["TV", "SEA", "Digital"],
        ...     controls=["Cospirit", "Concurence", "trend"],
        ... )
        >>> ia.get_coords()  # doctest: +NORMALIZE_WHITESPACE
        {'media': ['Digital', 'SEA', 'TV'],
         'controls': ['Concurence', 'Cospirit', 'trend'],
         'interaction_Concurence': ['Digital'],
         'interaction_Cospirit': ['Digital', 'SEA'],
         'interaction_TV': ['Digital,SEA']}
        """
        coords: dict[str, list[str]] = {}

        all_names = list(self.formulas.keys()) + self.media
        for term in sorted(self.get_all_interaction_terms()):
            seen_lhs: set[str] = set()
            term_coords: list[str] = []
            for name in sorted(all_names):
                if term not in self.parse_formula(name).terms:
                    continue
                lhs = self._get_lhs_label(name, term)
                if lhs not in seen_lhs:
                    seen_lhs.add(lhs)
                    term_coords.append(lhs)
            coords[f"interaction_{term}"] = sorted(term_coords)

        return coords

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_lhs_label(self, media_name: str, interact_var: str) -> str:
        """Return the coordinate label (LHS) for a (media, interact_var) pair.

        For a shared group the label is the comma-joined sorted names of all
        media in the group (e.g. ``"Digital,SEA"``); for an unshared term it
        is just *media_name*.
        """
        group = self._find_sharing_group(media_name, interact_var)
        if group is not None:
            _, *media_vars = group
            return ",".join(sorted(media_vars))
        return media_name

    def _find_sharing_group(
        self, media_name: str, interact_var: str
    ) -> tuple[str, ...] | None:
        """Find the sharing group for a (media, interaction variable) pair.

        Parameters
        ----------
        media_name : str
            Media variable name.
        interact_var : str
            Interaction variable name.

        Returns
        -------
        tuple[str, ...] | None
            The matching group tuple, or ``None`` if not found.
        """
        if self.is_shared_with is None:
            return None
        for group in self.is_shared_with:
            ivar, *media_vars = group
            if ivar == interact_var and media_name in media_vars:
                return group
        return None


if __name__ == "__main__":
    ia = Interaction(
        formulas={
            "TV": "1",
            "SEA": "1 + TV + Cospirit",
            "Digital": "0 + TV + Cospirit + Concurence",
            "Cospirit": "1",
            "Concurence": "0",
        },
        is_shared_with=[("TV", "Digital", "SEA")],
        media=["TV", "SEA", "Digital"],
        controls=["Cospirit", "Concurence", "trend"],
    )
    print(ia.get_unique_parameter_names())
    print(ia.get_coords())
