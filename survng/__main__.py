"""Start the SurvNG server, or run its host-local status command."""

import sys

from survng.app.__main__ import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        from survng.ctl import main as ctl_main

        raise SystemExit(ctl_main(sys.argv[1:]))
    main()
