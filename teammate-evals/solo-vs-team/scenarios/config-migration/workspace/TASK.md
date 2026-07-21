# Configuration migration repair

Repair `src/config_migration.py`. The service must safely load three historical
configuration formats, interpolate deployment variables, validate the canonical
form, and expose a recursively redacted public view. Keep the two public
function signatures unchanged.
