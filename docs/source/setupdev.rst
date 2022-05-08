Set up Development Environment
******************************

Local Environment
==================
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

VS Code Dev Container
======================

If you are a Visual Studio Code user, you may choose to develop inside a container. The benefit is the container comes with all necessary settings and dependencies configured. You will need `Docker <https://www.docker.com/>`_ installed in your system. You also need to have the `Remote - Containers <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_ extension enabled.

* Open the project in Visual Studio Code. in the status bar, select open in remote container.
  
It will perhaps take a few minutes the first time you build the container.
