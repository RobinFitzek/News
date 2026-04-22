{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.investment-monitor;

  # Libraries WeasyPrint needs at runtime (PDF export).
  # WeasyPrint >= 60 uses cffi bindings directly — no GObject typelibs needed.
  weasyPrintLibs = with pkgs; [
    pango cairo glib gdk-pixbuf harfbuzz fontconfig freetype
  ];
in {
  options.services.investment-monitor = {
    enable = mkEnableOption "AI Investment Monitor";

    appDir = mkOption {
      type    = types.path;
      example = "/home/robin/News";
      description = ''
        Absolute path to the cloned repository.
        The venv must already exist here (run setup.sh once first).
      '';
    };

    user = mkOption {
      type    = types.str;
      default = "investment-monitor";
      description = ''
        User account the service runs as.
        Set this to your own username if you want the service to share
        your home directory without extra permission juggling.
      '';
    };

    group = mkOption {
      type    = types.str;
      default = "investment-monitor";
    };

    port = mkOption {
      type    = types.port;
      default = 8443;
      description = "Port the Uvicorn web server listens on.";
    };

    environmentFile = mkOption {
      type    = types.nullOr types.path;
      default = null;
      example = "/etc/investment-monitor/secrets.env";
      description = ''
        Path to a file of KEY=VALUE pairs injected as environment variables.
        Keep ENCRYPTION_KEY, CSRF_SECRET_KEY, and API keys here so they
        survive rebuilds and stay out of the Nix store.
        See nixos/secrets.env.example for the required format.
        chmod 600 this file and own it by root.
      '';
    };

    tailscaleOnly = mkOption {
      type    = types.bool;
      default = true;
      description = ''
        When true, the port is only reachable via the tailscale0 interface.
        Set to false if you also want LAN access without Tailscale.
      '';
    };

    createUser = mkOption {
      type    = types.bool;
      default = true;
      description = ''
        Create a dedicated system user for the service.
        Set false when running as your own user account (user option set
        to your username) — NixOS won't try to create an account that
        already exists.
      '';
    };
  };

  config = mkIf cfg.enable {

    # --- User / group ----------------------------------------------------------
    users.users.${cfg.user} = mkIf cfg.createUser {
      isSystemUser = true;
      group        = cfg.group;
      home         = cfg.appDir;
      description  = "AI Investment Monitor service account";
    };

    users.groups.${cfg.group} = mkIf cfg.createUser {};

    # --- Firewall: Tailscale-only access ---------------------------------------
    # Blocks the port on every interface except tailscale0, so neither the
    # LAN nor the internet can reach the app directly.
    networking.firewall.interfaces."tailscale0".allowedTCPPorts =
      mkIf cfg.tailscaleOnly [ cfg.port ];

    # --- Systemd service -------------------------------------------------------
    systemd.services.investment-monitor = {
      description   = "AI Investment Monitor";
      documentation = [ "https://github.com/robinfitzek/news" ];

      # Wait for Tailscale to come up so the interface exists before we bind.
      after = [ "network-online.target" "tailscaled.service" ];
      wants = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        PYTHONUNBUFFERED = "1";
        WEB_HOST         = "0.0.0.0";
        WEB_PORT         = toString cfg.port;

        # WeasyPrint (PDF export) needs these shared libraries.
        # Without LD_LIBRARY_PATH the pip-installed weasyprint can't find
        # pango/cairo because NixOS doesn't use the standard /usr/lib paths.
        LD_LIBRARY_PATH = lib.makeLibraryPath weasyPrintLibs;
      };

      serviceConfig = {
        Type            = "simple";
        User            = cfg.user;
        Group           = cfg.group;
        WorkingDirectory = cfg.appDir;

        ExecStart = "${cfg.appDir}/venv/bin/python ${cfg.appDir}/main.py";

        # Inject secrets without touching the Nix store or .env files.
        EnvironmentFile = mkIf (cfg.environmentFile != null) cfg.environmentFile;

        Restart    = "on-failure";
        RestartSec = "10";

        StandardOutput    = "journal";
        StandardError     = "journal";
        SyslogIdentifier  = "investment-monitor";

        # Lightweight hardening — sensible for a homelab service without
        # locking down so hard that writes to appDir/core/data break.
        PrivateTmp           = true;
        NoNewPrivileges      = true;
        ProtectKernelTunables  = true;
        ProtectKernelModules   = true;
        ProtectControlGroups   = true;
        RestrictRealtime       = true;
        RestrictNamespaces     = true;
      };
    };
  };
}
