# Policy requirements

- A `Rule` effect is exactly `allow` or `deny`. Action and resource patterns use
  shell-style `*` wildcards with case-sensitive matching.
- Users may have several roles. Roles inherit all rules from their declared
  parents, transitively.
- Unknown user role names do not grant access. Unknown inherited role names and
  inheritance cycles are configuration errors and raise `ValueError` when the
  engine is created.
- Evaluate all matching rules across all assigned and inherited roles. Any
  matching deny overrides every allow. At least one allow is required; default
  is deny.
- Resource patterns may contain `{tenant}`. Replace it with the non-empty
  `context["tenant"]` value before matching. If no tenant is supplied, that rule
  cannot match. Substitution is literal: wildcard characters in a tenant value
  must not become policy wildcards.
- Engine construction must detach its internal role mapping from caller-owned
  dictionaries and rule lists.
