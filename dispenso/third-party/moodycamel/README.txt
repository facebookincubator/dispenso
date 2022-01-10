https://github.com/cameron314/concurrentqueue

commit 4cc3bee07b9ef252cfdfd3fddbdb476dc750179f (HEAD -> master, origin/master, origin/HEAD)
Author: Cameron <cameron@moodycamel.com>
Date:   Fri Jun 26 08:42:52 2020 -0400

    Fixed accidental reference being taken on new static const variable with g++ (issue #208)

commit 790db30df598bb45fdc63b571847160a2f9eb60b
Author: Cameron <cameron@moodycamel.com>
Date:   Wed Jun 24 15:05:32 2020 -0400

    Added MAX_SEMA_SPINS to traits to control semaphore spin-waiting (see issue #205)

commit 7912b95d041e658732e0732dd01794232a0a2892
Author: Cameron <cameron@moodycamel.com>
Date:   Wed Jun 24 14:43:30 2020 -0400

    Updated license comments to make extra-clear that the code is dual-licensed under the Boost terms (issue #207)

commit 38e6a6f0185a98c3aaf2a95aa109ba041221d527
Merge: 5c27828 de16a68
Author: Cameron <cameron@moodycamel.com>
Date:   Wed Apr 22 11:35:07 2020 -0400

    Merge pull request #200 from khuck/master
    
    Fixing check for address sanitizer

commit de16a68f453f9fcaec742111f5269ea7a3d4dcdf
Author: Kevin Huck <khuck@cs.uoregon.edu>
Date:   Wed Apr 22 08:33:09 2020 -0700

    Update concurrentqueue.h

commit a2fe6f5dfb12791a8ec4d8a60a03e42a9329461b
Author: Kevin Huck <khuck@cs.uoregon.edu>
Date:   Wed Apr 22 08:21:10 2020 -0700

    Fixing check for address sanitizer

commit 5c278285c247ff315059459762cb9ca15c8c4bcf
Merge: c8632bb da34fb9
Author: Cameron <cameron@moodycamel.com>
Date:   Tue Apr 21 19:46:29 2020 -0400

    Merge pull request #198 from graphicsMan/master
    
    Add macro to disable TSAN reporting on a per-function basis

commit da34fb977c0bae9120f4990e810324f7993537f4
Author: Brian Budge <bbudge@fb.com>
Date:   Thu Apr 16 09:41:12 2020 -0700

    Add comments about the purpose/usage of MOODYCAMEL_NO_TSAN

commit 75732bf18f0fe890c495656ac12594e97c838d42
Author: Brian Budge <bbudge@fb.com>
Date:   Thu Apr 16 08:32:09 2020 -0700

    Add a macro for per-function TSAN report suppression
