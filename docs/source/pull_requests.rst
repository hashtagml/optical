Pull Requests
*************

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
* Merge the PR. 

