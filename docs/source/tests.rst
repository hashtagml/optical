Writing tests
=============
Writing units tests is very important to ensure correctness of your changes and to make sure your changes are not breaking current behaviour unintensionally.

Below are the characteristics of a good unit test as mentioned `here <https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices#characteristics-of-a-good-unit-test>`_

* **Fast**: It is not uncommon for mature projects to have thousands of unit tests. Unit tests should take very little time to run. Milliseconds.
* **Isolated**: Unit tests are standalone, can be run in isolation, and have no dependencies on any outside factors such as a file system or database.
* **Repeatable**: Running a unit test should be consistent with its results, that is, it always returns the same result if you do not change anything in between runs.
* **Self-Checking**: The test should be able to automatically detect if it passed or failed without any human interaction.
* **Timely**: A unit test should not take a disproportionately long time to write compared to the code being tested. If you find testing the code taking a large amount of time compared to writing the code, consider a design that is more testable.

**Note**: All the tests will be run automatically through github actions whenever a pull request is raised and subsequest commits on top it.
