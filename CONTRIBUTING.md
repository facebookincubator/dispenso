# Contributing to dispenso
We want to make contributing to this project as easy and transparent as
possible.  There is a design ethos behind the library, so it is recommended to reach out via a GitHub 
issue on the project to discuss non-trivial changes you may wish to make.  These changes include, for 
example, wanting to change existing API, wanting to furnish a new utility, or wanting to change 
underlying behavior substantially.  Let's avoid situations where you put in a lot of hard work, only 
to have to change it substantially or get your pull request rejected.

## Our Development Process
This library has another home inside Facebook repos.  From there it is subjected to regular continuous integration testing on many platforms, and used by many projects.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Utilize clang-format.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style  
* 2 spaces for indentation rather than tabs
* 100 character line length
* Member variables have trailing underscore_
* BigCamelCase for classes and structs, and smallCamelCase for functions and variables (exception is if you are trying to match a substantial part of a standard library interface).
* [1TBS braces](https://en.wikipedia.org/wiki/Indentation_style#Variant:_1TBS_(OTBS))
* Most of all, try to be consistent with the surrounding code.  We have automated tools that will
  enforce clang-format style for some files (e.g. the C++ core) once we import your pull request
  into our internal code reviewing tools.

## License
By contributing to dispenso, you agree that your contributions will be licensed
under the LICENSE.md file in the root directory of this source tree.
