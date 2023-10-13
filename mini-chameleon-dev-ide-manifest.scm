;; Ce qui suit est un « manifeste » équivalent à la ligne de commande que vous avez passée.
;; Vous pouvez le sauvegarder dans un fichier qui vous pourrez passer aux commandes « guix »
;; qui acceptent l'option « --manifest » (ou « -m »).

(concatenate-manifests
  (list (specifications->manifest
          (list "emacs-bedrock-as-default"
                "clang-toolchain"))
        (package->development-manifest
          (specification->package "mini-chameleon"))))
