"""
Configuration classes for the MMM model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Interaction helpers
# ---------------------------------------------------------------------------

_TERM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALID_MODES = frozenset({"boost", "product"})


@dataclass(slots=True)
class InteractionFormula:
    """Parsed representation of a single-media interaction formula.

    A formula is a ``+``-separated list of terms. ``1`` marks the baseline
    as present, ``0`` marks it as explicitly absent (patsy/R convention) —
    exactly one of the two must appear. Any other identifier names a
    variable that modulates the channel.

    Each term may carry an explicit combination-mode suffix, ``:boost`` or
    ``:product``:

    * ``boost`` — modulates the channel's own beta multiplicatively:
      ``β_m · (1 + β_{m:Z} · Z)``.
    * ``product`` — contributes an independent additive term:
      ``β_m + β_{m:Z} · Z``.

    Without a suffix, the mode is inferred from the term's role: ``product``
    if the term is itself a media channel, ``boost`` otherwise. See
    :meth:`Interaction.get_interaction_mode`.

    Attributes
    ----------
    media_name : str
        Name of the media variable this formula belongs to.
    raw : str
        The original formula string.
    terms : list[str]
        Interaction variable names (every token other than ``"1"``/``"0"``).
    has_baseline : bool
        ``True`` when ``"1"`` appears in the formula, ``False`` when ``"0"``
        appears instead.
    term_modes : dict[str, str]
        ``{term: mode}`` for terms carrying an explicit ``:boost``/
        ``:product`` suffix; terms without one are absent from this dict.

    Raises
    ------
    ValueError
        If the formula contains unsupported syntax, an invalid mode suffix,
        a duplicated term, or both ``"1"`` and ``"0"`` (or neither).

    Examples
    --------
    >>> InteractionFormula(media_name="Y", raw="1 + Y2").terms
    ['Y2']
    >>> InteractionFormula(media_name="Y", raw="0 + Y2").has_baseline
    False
    >>> InteractionFormula(media_name="Y", raw="1 + C:product").term_modes
    {'C': 'product'}
    >>> InteractionFormula(media_name="Y", raw="1 + 0")
    Traceback (most recent call last):
        ...
    ValueError: Formula '1 + 0' cannot contain both '1' and '0'.
    >>> InteractionFormula(media_name="Y", raw="1 + Y2 + Y2")
    Traceback (most recent call last):
        ...
    ValueError: Duplicate term 'Y2' in formula '1 + Y2 + Y2'.
    >>> InteractionFormula(media_name="Y", raw="1 + C:weird")
    Traceback (most recent call last):
        ...
    ValueError: Invalid interaction mode 'weird' for term 'C' in formula
        '1 + C:weird'. Must be 'boost' or 'product'.
    """

    media_name: str
    raw: str
    terms: list[str] = field(init=False)
    has_baseline: bool = field(init=False)
    term_modes: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.terms, self.has_baseline, self.term_modes = self._parse(self.raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_term_token(tok: str, formula: str) -> tuple[str, str | None]:
        """Parse a single non-``1``/``0`` token into ``(name, mode)``.

        Parameters
        ----------
        tok : str
            The stripped token, e.g. ``"C"`` or ``"C:product"``.
        formula : str
            The full formula string, used for error messages.

        Returns
        -------
        tuple[str, str | None]
            The term name, and its explicit mode (``None`` if unspecified).

        Raises
        ------
        ValueError
            If the name is not a valid identifier, or the mode suffix is
            neither ``"boost"`` nor ``"product"``.
        """
        name, _, mode = tok.partition(":")
        name = name.strip()
        mode = mode.strip() if ":" in tok else None

        if not _TERM_RE.match(name):
            raise ValueError(
                f"Invalid token '{tok}' in formula '{formula}'. Each term must be "
                "'1', '0', or a valid Python identifier optionally suffixed with "
                "':boost' or ':product'."
            )
        if mode is not None and mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid interaction mode '{mode}' for term '{name}' in formula "
                f"'{formula}'. Must be 'boost' or 'product'."
            )
        return name, mode

    @staticmethod
    def _parse(formula: str) -> tuple[list[str], bool, dict[str, str]]:
        """Parse the formula string into interaction variable names and modes.

        Parameters
        ----------
        formula : str
            Formula string, e.g. ``"1 + Y2 + C:boost"``.

        Returns
        -------
        tuple[list[str], bool, dict[str, str]]
            A 3-tuple ``(terms, has_baseline, term_modes)`` where *terms* is
            the list of non-``"1"``/``"0"`` identifier tokens, *has_baseline*
            is ``True`` when ``"1"`` appears in the formula, and *term_modes*
            maps terms with an explicit ``:boost``/``:product`` suffix to
            that mode.

        Raises
        ------
        ValueError
            If any token is neither ``"1"``, ``"0"``, nor a valid Python
            identifier (optionally suffixed with a valid mode); if a term is
            duplicated; or if the formula contains both ``"1"`` and ``"0"``,
            or neither.
        """
        tokens = [t.strip() for t in formula.split("+")]
        has_baseline = False
        no_baseline = False
        terms: list[str] = []
        term_modes: dict[str, str] = {}
        seen: set[str] = set()

        for tok in tokens:
            if tok == "1":
                has_baseline = True
                continue
            if tok == "0":
                no_baseline = True
                continue

            name, mode = InteractionFormula._parse_term_token(tok, formula)
            if name in seen:
                raise ValueError(f"Duplicate term '{name}' in formula '{formula}'.")
            seen.add(name)
            terms.append(name)
            if mode is not None:
                term_modes[name] = mode

        if has_baseline and no_baseline:
            raise ValueError(f"Formula '{formula}' cannot contain both '1' and '0'.")
        if not has_baseline and not no_baseline:
            raise ValueError(
                f"Formula '{formula}' has no baseline. "
                "It must contain either '1' (baseline present) or '0' (no baseline)."
            )

        return terms, has_baseline, term_modes

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_default(self) -> bool:
        """Return ``True`` if the formula is the trivial ``"1"`` formula.

        Returns
        -------
        bool
            ``True`` when there are no interaction terms and the baseline
            is present (i.e. the formula is exactly ``"1"``).
        """
        return len(self.terms) == 0 and self.has_baseline


@dataclass(frozen=True, slots=True)
class SharingGroup:
    """A group of media channels sharing a single interaction coefficient.

    Declares that a specific interaction variable is shared across a set of
    media channels: instead of each channel getting its own
    ``"beta_interaction_{var}"`` coefficient value, they share one.

    Attributes
    ----------
    interact_var : str
        The interaction variable whose coefficient is shared.
    media : tuple[str, ...]
        The media channels sharing that coefficient (at least two).

    Raises
    ------
    ValueError
        If fewer than two media names are given, or if ``interact_var`` is
        empty (after stripping whitespace).

    Examples
    --------
    >>> SharingGroup(interact_var="TV", media=("Digital", "SEA"))
    SharingGroup(interact_var='TV', media=('Digital', 'SEA'))
    """

    interact_var: str
    media: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.media) < 2:
            raise ValueError(
                f"A sharing group must list at least two media names, got: {self.media}"
            )
        if not self.interact_var.strip():
            raise ValueError("Interaction variable name must be a non-empty string")

    @classmethod
    def coerce(cls, value: SharingGroup | tuple[str, ...]) -> SharingGroup:
        """Coerce a raw ``(interact_var, *media)`` tuple into a ``SharingGroup``.

        Parameters
        ----------
        value : SharingGroup | tuple[str, ...]
            Either a ``SharingGroup``, or a plain
            ``(interact_var, media_1, media_2, ...)`` tuple.

        Returns
        -------
        SharingGroup
            *value* unchanged if it already is a ``SharingGroup``, otherwise
            the tuple converted into one.
        """
        if isinstance(value, SharingGroup):
            return value
        interact_var, *media = value
        return cls(interact_var=interact_var, media=tuple(media))


@dataclass(slots=True)
class Interaction:
    """Centralized, validated definition of all media-variable interactions in an MMM.

    Each media channel can optionally have an interaction *formula* that
    describes how its effective coefficient is modulated by other media or
    control variables.  The mathematical model for a channel **Y** with
    formula ``"1 + Z"`` is:

    .. math::

        Y_{\\text{eff}} = \\beta_Y \\cdot (1 + \\beta_{Y:Z} \\cdot Z) \\cdot X_Y

    Validates configuration coherence; PyMC coordinate and parameter-name
    translation lives in :class:`InteractionCoordinates`.

    Parameters
    ----------
    formulas : dict[str, str], optional
        Mapping ``{media_name: formula_string}``.  Absent channels are
        treated as having the default ``"1"`` formula (no interaction).
    is_shared_with : list[SharingGroup] | list[tuple[str, ...]] | None, optional
        Each entry declares that a specific interaction variable is shared
        across a set of media channels. Accepts :class:`SharingGroup`
        instances, or plain ``(interact_var, *media)`` tuples (e.g.
        ``("TV", "Digital", "SEA")``), converted automatically.

        ``("TV", "Digital", "SEA")`` means the ``TV`` interaction
        coefficient is shared between ``Digital`` and ``SEA``. Other terms
        in those channels' formulas (e.g. ``Cospirit``) are not shared and
        each get their own parameter. A ``(media, interact_var)`` pair may
        belong to at most one sharing group.
    controls : list[str], optional
        Names of all control variables (every variable that is **not** a
        media channel).  A control not referenced in any formula term is
        treated as having a default interaction coefficient of ``1``
        (i.e. no modulation).  Use :meth:`get_default_controls` to
        retrieve these controls.

    Notes
    -----
    Each interaction variable has a combination mode, ``"boost"`` or
    ``"product"`` (see :class:`InteractionFormula`), defaulting to
    ``"product"`` for media channels and ``"boost"`` otherwise. A
    variable's mode must match across every formula that references it —
    once tagged explicitly in one, it must be tagged explicitly, with the
    same value, in all others. See :meth:`get_interaction_mode`.

    Attributes
    ----------
    formulas : dict[str, str]
        Raw formula strings keyed by media name.
    is_shared_with : list[SharingGroup] | None
        Sharing groups, or ``None`` when no sharing is requested.
    controls : list[str]
        Control variable names.

    Raises
    ------
    ValueError
        If a formula references an undefined term, if a sharing group
        references a media channel that does not have the corresponding
        interaction variable in its formula, if a ``(media, interact_var)``
        pair is claimed by more than one sharing group, or if an
        interaction variable's mode is inconsistent across the formulas
        that reference it.

    Examples
    --------
    >>> ia = Interaction(
    ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
    ...     is_shared_with=[("Y3", "Y1", "Y2")],
    ...     media=["Y1", "Y2"],
    ... )
    >>> ia.get_all_interaction_terms()
    {'Y3'}
    >>> ia.get_interaction_mode("Y3")  # not a media channel → defaults to boost
    'boost'
    """

    formulas: dict[str, str] = field(default_factory=dict)
    is_shared_with: list[SharingGroup] | list[tuple[str, ...]] | None = None
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
        if self.is_shared_with is not None:
            self.is_shared_with = [
                SharingGroup.coerce(group) for group in self.is_shared_with
            ]
        self._parsed = {
            name: InteractionFormula(media_name=name, raw=raw)
            for name, raw in self.formulas.items()
        }
        self._validate_all_terms_defined()
        self._validate_sharing_groups()
        self._validate_interaction_modes()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_sharing_groups(self) -> None:
        """Validate sharing groups and detect conflicting group membership.

        Every media channel listed in a sharing group must have the group's
        interaction variable in its own formula. Additionally, a given
        ``(media, interact_var)`` pair may only belong to a single sharing
        group; this is checked in the same pass.

        Raises
        ------
        ValueError
            If a media variable listed in a sharing group does not have
            that interaction variable in its formula, or if a
            ``(media, interact_var)`` pair is claimed by more than one
            sharing group.
        """
        if self.is_shared_with is None:
            return

        seen: dict[tuple[str, str], SharingGroup] = {}
        for group in self.is_shared_with:
            assert isinstance(group, SharingGroup)
            for media_name in group.media:
                parsed = self._parsed.get(media_name)
                if parsed is None or group.interact_var not in parsed.terms:
                    raise ValueError(
                        f"Sharing group {group}: media variable '{media_name}' does "
                        f"not have '{group.interact_var}' in its formula "
                        f"(formula: '{self.formulas.get(media_name, '1')}')."
                    )
                key = (media_name, group.interact_var)
                if key in seen:
                    raise ValueError(
                        f"Conflicting sharing groups: media '{media_name}' and "
                        f"interaction variable '{group.interact_var}' belong to "
                        f"both {seen[key]} and {group}."
                    )
                seen[key] = group

    def _validate_interaction_modes(self) -> None:
        """Check that every interaction variable resolves to one consistent mode.

        Raises
        ------
        ValueError
            If a variable is tagged explicitly in some formulas but not
            others that reference it, or if formulas disagree on its
            explicit mode, or if a variable has no explicit
            ``:boost``/``:product`` tag in any formula.
        """
        occurrences: dict[str, list[tuple[str, str | None]]] = {}
        for media_name, formula in self._parsed.items():
            for term in formula.terms:
                occurrences.setdefault(term, []).append(
                    (media_name, formula.term_modes.get(term))
                )

        implicit_terms: list[str] = []
        for term, entries in occurrences.items():
            explicit = [(m, mode) for m, mode in entries if mode is not None]
            if not explicit:
                implicit_terms.append(term)
                continue
            if len(explicit) != len(entries):
                missing = sorted(m for m, mode in entries if mode is None)
                raise ValueError(
                    f"Interaction variable '{term}' is tagged ':boost'/':product' "
                    f"in some formulas but not others that reference it. Add the "
                    f"tag in: {missing}."
                )
            modes = {mode for _, mode in explicit}
            if len(modes) > 1:
                raise ValueError(
                    f"Interaction variable '{term}' has conflicting explicit "
                    f"modes across formulas: {sorted(explicit)}."
                )

        if implicit_terms:
            hints = ", ".join(
                f"'{term}' (inferred: '{self.get_interaction_mode(term)}')"
                for term in sorted(implicit_terms)
            )
            raise ValueError(
                f"Interaction variable(s) {hints} have no explicit ':boost'/"
                "':product' tag. Tag them explicitly, e.g. 'TV:product', to make "
                "the mode independent of the variable's media/control role."
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

    def get_all_interaction_terms(self) -> set[str]:
        """Return the set of every distinct interaction variable used in any formula.

        Returns
        -------
        set[str]
            Union of all non-``"1"``/``"0"`` tokens across all registered
            formulas.

        Examples
        --------
        >>> ia = Interaction(formulas={"Y1": "1 + Y3", "Y2": "1 + C", "Y3": "1", "C": "1"})
        >>> ia.get_all_interaction_terms() == {"Y3", "C"}
        True
        """
        terms: set[str] = set()
        for parsed in self._parsed.values():
            terms.update(parsed.terms)
        return terms

    def get_interaction_mode(self, interact_var: str) -> str:
        """Return the combination mode for an interaction variable.

        Returns its explicit ``:boost``/``:product`` tag if any formula sets
        one, otherwise ``"product"`` if *interact_var* is a media channel,
        else ``"boost"``.

        Parameters
        ----------
        interact_var : str
            Interaction variable name.

        Returns
        -------
        str
            Either ``"boost"`` or ``"product"``.

        Examples
        --------
        >>> ia = Interaction(formulas={"Y1": "1 + Y2", "Y2": "1"}, media=["Y1", "Y2"])
        >>> ia.get_interaction_mode("Y2")  # Y2 is a media channel → product
        'product'
        >>> ia = Interaction(
        ...     formulas={"Y1": "1 + C:product", "C": "1"}, media=["Y1"], controls=["C"])
        >>> ia.get_interaction_mode("C")  # C is a control but explicitly overridden
        'product'
        """
        for formula in self._parsed.values():
            mode = formula.term_modes.get(interact_var)
            if mode is not None:
                return mode
        return "product" if interact_var in self.media else "boost"

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
        mentioned = self.get_all_interaction_terms() & set(self.controls)
        return sorted(set(self.controls) - mentioned)


@dataclass(slots=True)
class InteractionCoordinates:
    """Translate a validated :class:`Interaction` into PyMC coordinates.

    Handles parameter naming, shared-group resolution, and ``coords`` dict
    construction for ``pm.Model``. Assumes *interaction* has already been
    validated — sharing-group conflicts are :class:`Interaction`'s
    responsibility, not this class's.

    Parameters
    ----------
    interaction : Interaction
        An already-validated interaction configuration.

    Examples
    --------
    >>> ia = Interaction(
    ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
    ...     is_shared_with=[("Y3", "Y1", "Y2")],
    ...     media=["Y1", "Y2"],
    ... )
    >>> coords = InteractionCoordinates(ia)
    >>> coords.get_parameter_name("Y3")
    'beta_interaction_Y3'
    >>> coords.get_lhs_index("Y1", "Y3")   # "Y1,Y2" is index 0 in interaction_Y3
    0
    """

    interaction: Interaction
    _sharing_index: dict[tuple[str, str], SharingGroup] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self._sharing_index = {}
        if self.interaction.is_shared_with is None:
            return
        for group in self.interaction.is_shared_with:
            assert isinstance(group, SharingGroup)
            for media_name in group.media:
                self._sharing_index[(media_name, group.interact_var)] = group

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        >>> InteractionCoordinates(ia).get_parameter_name("Y3")
        'beta_interaction_Y3'

        >>> ia = Interaction(
        ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        ...     is_shared_with=[("Y3", "Y1", "Y2")],
        ...     media=["Y1", "Y2"],
        ... )
        >>> InteractionCoordinates(ia).get_parameter_name("Y3")
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
        >>> InteractionCoordinates(ia).get_lhs_index("Y1", "Y3")  # shared → LHS is "Y1,Y2", index 0
        0
        >>> ia2 = Interaction(
        ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        ...     media=["Y1", "Y2"],
        ... )
        >>> InteractionCoordinates(ia2).get_lhs_index("Y2", "Y3")  # unshared → LHS is "Y2", index 1
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
        ...     formulas={"Y1": "1 + Y3 + C", "Y2": "1 + Y3", "Y3": "1", "C": "1"},
        ...     is_shared_with=[("Y3", "Y1", "Y2")],
        ...     media=["Y1", "Y2"],
        ...     controls=["C"],
        ... )
        >>> InteractionCoordinates(ia).resolve_shared_groups("Y1")
        {'Y3': 'beta_interaction_Y3', 'C': 'beta_interaction_C'}
        """
        parsed = self.interaction.parse_formula(media_name)
        return {term: self.get_parameter_name(term) for term in parsed.terms}

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
        ...     formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        ...     is_shared_with=[("Y3", "Y1", "Y2")],
        ...     media=["Y1", "Y2"],
        ... )
        >>> InteractionCoordinates(ia).get_unique_parameter_names()
        {'beta_interaction_Y3'}
        """
        return {
            f"beta_interaction_{term}"
            for term in self.interaction.get_all_interaction_terms()
        }

    def get_coords(self) -> dict[str, list[str]]:
        """Return xarray-compatible coordinate lists for interaction parameters.

        Keys and their meaning:

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
        >>> InteractionCoordinates(ia).get_coords()  # doctest: +NORMALIZE_WHITESPACE
        {'interaction_Concurence': ['Digital'],
         'interaction_Cospirit': ['Digital', 'SEA'],
         'interaction_TV': ['Digital,SEA']}
        """
        coords: dict[str, list[str]] = {}

        all_names = list(self.interaction.formulas.keys()) + self.interaction.media
        for term in sorted(self.interaction.get_all_interaction_terms()):
            seen_lhs: set[str] = set()
            term_coords: list[str] = []
            for name in sorted(all_names):
                if term not in self.interaction.parse_formula(name).terms:
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
            return ",".join(sorted(group.media))
        return media_name

    def _find_sharing_group(
        self, media_name: str, interact_var: str
    ) -> SharingGroup | None:
        """Find the sharing group for a (media, interaction variable) pair.

        Parameters
        ----------
        media_name : str
            Media variable name.
        interact_var : str
            Interaction variable name.

        Returns
        -------
        SharingGroup | None
            The matching group, or ``None`` if not found.
        """
        return self._sharing_index.get((media_name, interact_var))


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
    thecoords = InteractionCoordinates(ia)
    print(thecoords.get_unique_parameter_names())
    print(thecoords.get_coords())
