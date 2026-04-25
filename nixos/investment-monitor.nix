{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.investment-monitor;

  # Native libraries WeasyPrint and other pip packages need at runtime.
  weasyPrintLibs = with pkgs; [
    pango cairo glib gdk-pixbuf harfbuzz fontconfig freetype
  ];

  # Resolved Python binary: Nix-managed package takes priority over the venv.
  pythonBin = if cfg.pythonPackage != null
              then "${cfg.pythonPackage}/bin/python"
              else "${cfg.appDir}/venv/bin/python";

in {
  options.services.investment-monitor = {

    enable = mkEnableOption "AI Investment Monitor";

    # ── Core paths ───────────────────────────────────────────────────────────────

    appDir = mkOption {
      type    = types.path;
      example = "/srv/investment-monitor";
      description = ''
        Absolute path to the cloned repository. The venv must already exist
        here (run ./setup.sh once) unless pythonPackage is set.
        Tip: keep a separate data volume and symlink data/ and logs/ into it
        so a `git pull` + `nixos-rebuild switch` replaces the app without
        touching persistent state.
      '';
    };

    pythonPackage = mkOption {
      type        = types.nullOr types.package;
      default     = null;
      example     = literalExpression
        "inputs.investment-monitor.packages.\${system}.pythonEnv";
      description = ''
        When set, use this Nix-built Python package instead of the virtualenv
        inside appDir. The flake exposes packages.pythonEnv which covers the
        most common dependencies; remaining packages (google-genai, ts2vg, …)
        must still be installed via pip or baked into your own derivation.
      '';
    };

    # ── Identity ─────────────────────────────────────────────────────────────────

    user = mkOption {
      type    = types.str;
      default = "investment-monitor";
      description = ''
        Unix user the service runs as. Set to your own username if you want
        the service to share your home directory without extra permission work.
        When doing so, also set createUser = false.
      '';
    };

    group = mkOption {
      type    = types.str;
      default = "investment-monitor";
    };

    createUser = mkOption {
      type    = types.bool;
      default = true;
      description = ''
        Create a dedicated system user and group for the service.
        Set false when running as an existing user account.
      '';
    };

    # ── Network ──────────────────────────────────────────────────────────────────

    port = mkOption {
      type    = types.port;
      default = 8443;
      description = "Port Uvicorn listens on.";
    };

    host = mkOption {
      type    = types.str;
      default = "0.0.0.0";
      description = ''
        Address Uvicorn binds to. The default (0.0.0.0) combined with
        tailscaleOnly = true is safe: the firewall blocks all non-Tailscale
        traffic. Set to "127.0.0.1" when using the nginx reverse proxy option
        so Uvicorn is not reachable directly.
      '';
    };

    tailscaleOnly = mkOption {
      type    = types.bool;
      default = true;
      description = ''
        Allow the port only on the tailscale0 interface via the firewall.
        Disable if you need direct LAN access without Tailscale.
      '';
    };

    openFirewall = mkOption {
      type    = types.bool;
      default = false;
      description = ''
        Open cfg.port on all interfaces. Only useful when tailscaleOnly = false
        and you want plain LAN access alongside (or instead of) Tailscale.
      '';
    };

    # ── Secrets ──────────────────────────────────────────────────────────────────

    environmentFile = mkOption {
      type    = types.nullOr types.path;
      default = null;
      example = "/etc/investment-monitor/secrets.env";
      description = ''
        Path to a KEY=VALUE file injected as environment variables.
        Keep ENCRYPTION_KEY, CSRF_SECRET_KEY, and API keys here so they
        survive rebuilds and never enter the Nix store.
        chmod 600 and chown root this file.
        See nixos/secrets.env.example for the required format.
      '';
    };

    # ── Frontend ─────────────────────────────────────────────────────────────────

    autoBuildFrontend = mkOption {
      type    = types.bool;
      default = false;
      description = ''
        Run `npm ci && npm run build` in the frontend/ directory before each
        service start. Enable this so that `git pull` + `nixos-rebuild switch`
        automatically rebuilds the React bundle without a manual step.
        Adds ~30–60 s to startup time on the first run after an update.
      '';
    };

    # ── Nginx reverse proxy ───────────────────────────────────────────────────────

    nginx = {
      enable = mkOption {
        type    = types.bool;
        default = false;
        description = ''
          Configure Nginx as a reverse proxy in front of Uvicorn.
          Nginx handles TLS termination; Uvicorn stays on plain HTTP.
          When enabled, consider setting host = "127.0.0.1" and
          tailscaleOnly = false so only Nginx is externally reachable.
        '';
      };

      hostname = mkOption {
        type    = types.str;
        default = "investment-monitor.local";
        description = ''
          Virtual host Nginx listens on. Use a real domain when useACME = true.
        '';
      };

      useACME = mkOption {
        type    = types.bool;
        default = false;
        description = ''
          Obtain a Let's Encrypt certificate via ACME.
          Requires the hostname to be publicly reachable on port 80.
          When false, configure TLS manually or rely on a local CA / mkcert.
        '';
      };
    };
  };

  # ── Implementation ─────────────────────────────────────────────────────────────

  config = mkIf cfg.enable {

    # ── User / group ────────────────────────────────────────────────────────────
    users.users.${cfg.user} = mkIf cfg.createUser {
      isSystemUser = true;
      group        = cfg.group;
      home         = cfg.appDir;
      description  = "AI Investment Monitor service account";
    };
    users.groups.${cfg.group} = mkIf cfg.createUser {};

    # ── Firewall ────────────────────────────────────────────────────────────────
    networking.firewall.interfaces."tailscale0".allowedTCPPorts =
      mkIf cfg.tailscaleOnly [ cfg.port ];

    networking.firewall.allowedTCPPorts =
      mkIf (cfg.openFirewall && !cfg.tailscaleOnly) [ cfg.port ];

    # ── Nginx (optional) ────────────────────────────────────────────────────────
    services.nginx = mkIf cfg.nginx.enable {
      enable = true;
      recommendedProxySettings = true;
      recommendedGzipSettings  = true;
      recommendedOptimisation   = true;
      recommendedTlsSettings    = true;

      virtualHosts.${cfg.nginx.hostname} = {
        enableACME = cfg.nginx.useACME;
        forceSSL   = cfg.nginx.useACME;
        locations."/" = {
          proxyPass       = "http://127.0.0.1:${toString cfg.port}";
          proxyWebsockets = true;
        };
      };
    };

    # ── Systemd service ─────────────────────────────────────────────────────────
    systemd.services.investment-monitor = {
      description   = "AI Investment Monitor";
      documentation = [ "https://github.com/robinfitzek/news" ];

      after = [
        "network-online.target"
        "tailscaled.service"
      ];
      wants    = [ "network-online.target" ];
      # Hard-require Tailscale when tailscaleOnly is set so the service fails
      # fast instead of starting unreachable.
      requires = mkIf cfg.tailscaleOnly [ "tailscaled.service" ];
      wantedBy = [ "multi-user.target" ];

      # Add nodejs to PATH so the npm ExecStartPre commands resolve correctly.
      path = lib.optionals cfg.autoBuildFrontend [ pkgs.nodejs_22 ];

      environment = {
        PYTHONUNBUFFERED = "1";
        WEB_HOST         = cfg.host;
        WEB_PORT         = toString cfg.port;
        LD_LIBRARY_PATH  = lib.makeLibraryPath weasyPrintLibs;
      };

      serviceConfig = {
        Type             = "simple";
        User             = cfg.user;
        Group            = cfg.group;
        WorkingDirectory = cfg.appDir;

        # Rebuild the React bundle before start when autoBuildFrontend is on.
        # `npm ci` installs/updates node_modules, then `npm run build` outputs
        # to frontend/dist which FastAPI serves as static files.
        ExecStartPre = lib.optionals cfg.autoBuildFrontend [
          "${pkgs.nodejs_22}/bin/npm --prefix ${cfg.appDir}/frontend ci"
          "${pkgs.nodejs_22}/bin/npm --prefix ${cfg.appDir}/frontend run build"
        ];

        ExecStart = "${pythonBin} ${cfg.appDir}/main.py";

        EnvironmentFile = mkIf (cfg.environmentFile != null) cfg.environmentFile;

        Restart         = "on-failure";
        RestartSec      = "10s";
        TimeoutStartSec = "120";   # allow time for npm build on first update

        StandardOutput   = "journal";
        StandardError    = "journal";
        SyslogIdentifier = "investment-monitor";

        # ── Systemd hardening ──────────────────────────────────────────────────
        # ProtectSystem=strict makes /usr, /boot, /etc read-only; the service
        # can still write to appDir (data/, logs/) via ReadWritePaths.
        ProtectSystem   = "strict";
        ReadWritePaths  = [ cfg.appDir ];

        # /home and /root are read-only; ReadWritePaths overrides for appDir
        # even if appDir lives inside /home.
        ProtectHome     = "read-only";

        PrivateTmp      = true;
        PrivateUsers    = true;   # UID/GID namespace isolation

        NoNewPrivileges           = true;
        CapabilityBoundingSet     = "";       # no Linux capabilities needed
        AmbientCapabilities       = "";

        ProtectKernelTunables     = true;
        ProtectKernelModules      = true;
        ProtectKernelLogs         = true;
        ProtectControlGroups      = true;
        ProtectHostname           = true;
        ProtectClock              = true;

        LockPersonality           = true;
        RestrictRealtime          = true;
        RestrictNamespaces        = true;
        RestrictSUIDSGID          = true;

        # Allow only the syscall groups a well-behaved network service needs.
        SystemCallFilter          = "@system-service";
        SystemCallArchitectures   = "native";  # block 32-bit compat layer

        # Restrict to IP (v4 + v6) and Unix sockets; no Netlink, Bluetooth, etc.
        RestrictAddressFamilies   = "AF_INET AF_INET6 AF_UNIX";
      };
    };
  };
}
