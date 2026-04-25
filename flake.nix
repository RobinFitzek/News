{
  description = "AI Investment Monitor — reproducible NixOS deployment";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    let
      # The NixOS module — import into your host flake like:
      #
      #   inputs.investment-monitor.url = "github:robinfitzek/news";
      #
      #   nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      #     modules = [
      #       ./configuration.nix
      #       inputs.investment-monitor.nixosModules.default
      #     ];
      #   };
      module = ./nixos/investment-monitor.nix;
    in
    {
      # ── NixOS module ───────────────────────────────────────────────────────────
      nixosModules.investment-monitor = module;
      nixosModules.default            = module;

      # ── Per-architecture outputs ───────────────────────────────────────────────
    } // flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Native .so files that pip-installed packages (WeasyPrint, cryptography)
        # cannot find on NixOS without explicit LD_LIBRARY_PATH.
        nativeLibs = with pkgs; [
          pango cairo glib gdk-pixbuf harfbuzz fontconfig freetype
          libffi openssl zlib
        ];

        # Python environment with packages available in nixpkgs.
        # Packages not yet in nixpkgs (google-genai, ts2vg, pywebpush, …)
        # are installed by setup.sh into the virtualenv; they pick up
        # LD_LIBRARY_PATH from the systemd unit automatically.
        pythonEnv = pkgs.python311.withPackages (ps: [
          ps.fastapi
          ps.uvicorn
          ps.jinja2
          ps.requests
          ps.pandas
          ps.cryptography
          ps.passlib
          ps.aiofiles
          ps.apscheduler
          ps.aiosqlite
          ps.itsdangerous
          ps.networkx
          ps.statsmodels
          ps.scikit-learn
          ps.weasyprint
          ps.vaderSentiment
          ps.pyotp
        ]);

      in {
        # `nix build .#pythonEnv` produces a Python interpreter with common deps.
        # Point the NixOS module's pythonPackage option at this to avoid
        # managing a virtualenv manually.
        packages.pythonEnv = pythonEnv;
        packages.default   = pythonEnv;

        # `nix develop` drops into a shell with system libs + Python + Node.
        # Run ./setup.sh here to install remaining pip packages into the venv.
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.nodejs_22   # frontend build (npm run build)
            pkgs.git
          ] ++ nativeLibs;

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}:''${LD_LIBRARY_PATH:-}"
            echo "Investment Monitor dev shell ready."
            echo "  ./setup.sh   — create/update the virtualenv"
            echo "  cd frontend && npm ci && npm run build   — build the React bundle"
            echo "  python main.py   — start the server"
          '';
        };
      }
    );
}
