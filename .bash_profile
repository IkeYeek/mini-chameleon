# -*- mode: sh -*-

##############
# guix setup #
##############

# guix binary (where to find guix binary)
GUIX_PROFILE="$HOME/.config/guix/current"
. "$GUIX_PROFILE/etc/profile"

# guix profile (where to find packages installed with `guix install`)
GUIX_PROFILE="$HOME/.guix-profile"
. "$GUIX_PROFILE/etc/profile"

# Bash completion
# 1. You need to install the bash-completion https://guix.gnu.org/manual/en/html_node/Application-Setup.html#Locales-1 as follows:
# guix install bash-completion
# 2. You need 2 to uncomment the following line and fix your login (not agullo)
# . /var/guix/profiles/per-user/agullo/current-guix/etc/bash_completion.d/guix

# Locales
# You need to install the locales https://guix.gnu.org/manual/en/html_node/Application-Setup.html#Locales-1 as follows:
# guix install glibc-locales
export GUIX_LOCPATH=$HOME/.guix-profile/lib/locale

# Cerfificates for SSH (hence GIT) and R
# You need to install nss-certs https://guix.gnu.org/manual/en/html_node/Application-Setup.html#X_002e509-Certificates-1 as follows:
# guix install nss-certs
# SSH (hence GIT)
export SSL_CERT_DIR="$HOME/.guix-profile/etc/ssl/certs"
export SSL_CERT_FILE="$HOME/.guix-profile/etc/ssl/certs/ca-certificates.crt"
export GIT_SSL_CAINFO="$SSL_CERT_FILE"
# R
export CURL_CA_BUNDLE="$HOME/.guix-profile/etc/ssl/certs/ca-certificates.crt"


