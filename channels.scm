(list (channel
       (name 'guix-hpc)
       (url "https://gitlab.inria.fr/guix-hpc/guix-hpc.git")
       (branch "master")
       (commit
        "5df6c7b8881be1ffcdcbda5c41032f35c2e37f10"))
      (channel
       (name 'guix-hpc-non-free)
       (url "https://gitlab.inria.fr/guix-hpc/guix-hpc-non-free.git")
       (branch "master")
       (commit
        "07d368150436d64eded315f26a3576ee3bdbc160"))
      (channel
       (name 'guix)
       (url "https://git.savannah.gnu.org/git/guix.git")
       (branch "master")
       (commit
        "31708431c53524f05e6a0c9fed920cb773e7dd21")
       (introduction
        (make-channel-introduction
         "9edb3f66fd807b096b48283debdcddccfea34bad"
         (openpgp-fingerprint
          "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA")))))
