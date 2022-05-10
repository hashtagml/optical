..
    Adapted from https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62

###########################
Contributing to optical 🙏
###########################

We love your input! We want to make contributing to optical as easy and transparent as possible, whether it is:

* Reporting a bug 🐛
* Submitting a fix 🔧
* Proposing new features 🚀
* Becoming a maintainer 📌
* A generic discussion  💬

We use `Github↩ <https://github.com/hashtagml/optical>`_ to host code, to track issues and feature requests, as well as accept pull requests.


Issues
======

* Please create an issue if something is broken or could be improved with the existing code or you want to request a new feature.
* Consider adding the following details to your issue:
  
    * A quick summary and/or background
    * Steps to reproduce the behaviour
  
        * Be specific!
        * Provide appropriate code snippets if applicable.
        * If it's regarding a traceback, please provide the complete body of the traceback in the description.
    * Expected behaviour
    * Observed behaviour
    * Notes (possibly including why you think this might be happening, or things you've tried that didn't work)


Set up Development Environment
==============================

**Local Environment**

* Fork the repository
* We use `poetry↩ <https://python-poetry.org/>`_ to manage our dependencies and handle packaging requirements. Install ``poetry``

  .. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -

* Create a virtual environment (we prefer ``conda`` for that job)
  
  .. code-block:: bash

    conda create -n optical python=3.8 pip

* Install the dependencies and the project in editable mode
  
  .. code-block:: bash

    cd optical
    poetry install


**VSCode dev container**

If you are a Visual Studio Code user, you may choose to develop inside a container. The benefit is the container comes with all necessary settings and dependencies configured. You will need `Docker <https://www.docker.com/>`_ installed in your system. You also need to have the `Remote - Containers <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_ extension enabled.

* Open the project in Visual Studio Code. in the status bar, select open in remote container.
  
It will perhaps take a few minutes the first time you build the container.


Coding requirements
===================

* Please work on a separate branch while working on a feature or a bug. Name your branches appropriately, Examples: ``feature/yet_another_feature``, ``bugfix/annoying_bug``, ``docs/fix_typo``
* Make changes as required. Handle any exceptions. Please keep the changes minimal and your code clean.
* Use a consistent coding style. Please make appropriate use of docstrings(we follow `Google style docstring <https://google.github.io/styleguide/pyguide.html>`_ ), so that other can understand your code easily.
* Please write tests to cover the changes you made. You can find how to write tests in Tests section below.
* Write clear commit messages.
* Create/modify documentation if required and update changelog.
* Raise a pull request.


Writing tests
=============

Writing units tests is very important to ensure correctness of your changes and to make sure your changes are not breaking current behaviour unintentionally. 

If you are writing a new feature make sure you write unit tests for the same. We use `pytest <https://docs.pytest.org/en/7.1.x/>`_ for writing our test cases.

Here are a few characteristics of a good unit test. 

* **Fast**: It is not uncommon for mature projects to have thousands of unit tests. Unit tests should take very little time to run. Milliseconds.
* **Isolated**: Unit tests are standalone, can be run in isolation, and have no dependencies on any outside factors such as a file system or database.
* **Repeatable**: Running a unit test should be consistent with its results, that is, it always returns the same result if you do not change anything in between runs.
* **Self-Checking**: The test should be able to automatically detect if it passed or failed without any human interaction.
* **Timely**: A unit test should not take a disproportionately long time to write compared to the code being tested. If you find testing the code taking a large amount of time compared to writing the code, consider a design that is more testable.

..note::
  All the tests will be run automatically through github actions whenever a pull request is raised and subsequent commits on top it.

..tip::
  You can read more about standard practices about writing unit test in `Microsoft docs <https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices#characteristics-of-a-good-unit-test>`_.

Documentation
==============

Please add or modify documentation supporting the changes you have made. You can test the documentation locally using:
   
.. code-block:: bash
    
    tox -e docs #To use existing environment

Optionally, you can pass a tag ``--recreate`` to the above command in case you want to run a fresh build. If the build is successful, you can find your documentation under ``docs/build/html``.



Pull requests
=============

* Before raising pull request please ensure your branch is up to date with ``main`` branch. Others might have merged new changes to ``main`` after you started working on your branch.
* 
  Typical steps are listed below

    .. code-block:: bash

        #Assuming currently you are on your_branch
        git checkout main
        git pull --rebase
        git checkout your_branch
        git rebase main # Resolve any conflicts
        git push --force origin your_branch

* Raise a pull request with proper heading and description. Description should contain why this PR is being raised and what's included in the PR. You can always raise `draft PR <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`_ if your work is still in progress. Choose reviewers
* Fix any issues in the tests and resolve comments/changes from reviewers
* Maintainer will merge the PR. 

