# -*- mode: sh -*-

########
# guix #
########

export GUIX_LOCPATH=$HOME/.guix-profile/lib/locale

GUIX_PROFILE="$HOME/.config/guix/current"
. "$GUIX_PROFILE/etc/profile"

GUIX_PROFILE="$HOME/.guix-profile"
. "$GUIX_PROFILE/etc/profile"

# bash completion with guix
# 1. install bash-completion (~guix install bash-completion~)
# 2. uncomment the following line and fix your login (not agullo)
# . /var/guix/profiles/per-user/agullo/current-guix/etc/bash_completion.d/guix

