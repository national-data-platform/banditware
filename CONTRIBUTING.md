# Contributing to BanditWare

Before you start writing code, please get your environment set up.

1. Make sure you have all of the necessary packages installed and the code runs smoothly.
2. Setting up the formatter
   - If you are using VS Code, install the [Black Formatter Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter).
     1. Add the following to your VS Code's `settings.json` file to automatically format on save:
     ```
     "[python]": {
         "editor.defaultFormatter": "ms-python.black-formatter",
         "editor.formatOnSave": true
     },
     ```
   - If you are not using VS Code, install the [Black Formatter](https://github.com/psf/black) and run it on all updated files before committing anything.
3. Create a new branch, make any changes on that branch, then Pull Request that branch into `main`.
   - If there area any unformatted python files in your pull request, a github action will automatically run and commit an updated version of those files on your branch.
   - Make sure to run `git pull` after creating your pull request so you have those format changes locally as well.
