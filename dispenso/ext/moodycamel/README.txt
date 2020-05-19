https://github.com/cameron314/concurrentqueue

ommit 5c278285c247ff315059459762cb9ca15c8c4bcf (HEAD -> master, origin/master, origin/HEAD)
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
    
    Details: Add MOODYCAMEL_NO_TSAN macro, which compiles to nothing when not in TSAN mode, and which compiles to an attribute suppressing TSAN reporting on a per-function basis when in TSAN mode.  Additionally, apply this to ExplicitProducer::enqueue_bulk, which was triggering TSAN reports from my project.
    
    Tested: Copied code into my project and re-ran the unit test that was previously failing.  Code runs cleanly now.

commit c8632bbba2f0c95b0576097cd77190dc037e9eb1
Author: Cameron <cameron@moodycamel.com>
Date:   Thu Apr 16 08:28:54 2020 -0400

    Bug fix for alignment syntax on VS2013 (issue #194)
