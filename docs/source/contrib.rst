..
    Adapted from https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62

########################
Contributing to Optical
########################

We love your input! We want to make contributing to optical as easy and transparent as possible, whether it's:

* Reporting a bug
* Discussing the current state of the code
* Submitting a fix
* Proposing new features
* Becoming a maintainer

**We Develop with Github**

We use github to host code, to track issues and feature requests, as well as accept pull requests.


Issues
======

* Please create an issue if something is not working with the existing code or want to request a new feature.
* Great Bug Reports tend to have:
  
    * A quick summary and/or background
    * Steps to reproduce
  
        * Be specific!
        * Give sample code if you can.
    * What you expected would happen
    * What actually happens
    * Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)


Set up Development Environment
==============================

**Local Environment**

* Fork the repo
* Install ``poetry``

  .. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -

* Create a virtual Environment
  
  .. code-block:: bash

    conda create -n optical python=3.8 pip

* Install the dependencies and the project in editable mode
  
  .. code-block:: bash

    poetry install


**VS Code Dev Container**

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

Writing units tests is very important to ensure correctness of your changes and to make sure your changes are not breaking current behaviour unintensionally.

Below are the characteristics of a good unit test as mentioned `here <https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices#characteristics-of-a-good-unit-test>`_

* **Fast**: It is not uncommon for mature projects to have thousands of unit tests. Unit tests should take very little time to run. Milliseconds.
* **Isolated**: Unit tests are standalone, can be run in isolation, and have no dependencies on any outside factors such as a file system or database.
* **Repeatable**: Running a unit test should be consistent with its results, that is, it always returns the same result if you do not change anything in between runs.
* **Self-Checking**: The test should be able to automatically detect if it passed or failed without any human interaction.
* **Timely**: A unit test should not take a disproportionately long time to write compared to the code being tested. If you find testing the code taking a large amount of time compared to writing the code, consider a design that is more testable.

**Note**: All the tests will be run automatically through github actions whenever a pull request is raised and subsequest commits on top it.

Building Docs
==============

Add/Modify documentation as required and test these changes on your local or remote container using below command 
   
.. code-block:: bash
    
    tox -e docs #To use existing environment
    tox --recreate -e docs #To recreate the environment

Once docs build is successful, you can find built pages under ``docs/build``.


Pull Requests
=============

* Before raising pull request please make sure you branch is up to date with main branch. Others might have merged new changes to main after you started working on your branch.
  Typical steps are listed below

    .. code-block:: bash

        #Assuming currently you are on your_branch
        git checkout main
        git pull --rebase
        git checkout your_branch
        git rebase main # Resolve any conflicts
        git push --force origin your_branch

* Raise a pull request with proper heading and description. Description should contain why this PR is being raised and what's included in the PR. You can alwasy raise `draft PR <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`_ if your work is still in progress. Choose reviewers
* Fix any issues in the tests and resolve comments/changes from reviewers
* Maintainer will merge the PR. 
