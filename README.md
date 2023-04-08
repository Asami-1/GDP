# GDP
Repository related to Cranfield's AAI MSCs GDP

# Using git 

Clone the repo :

```console
git clone https://github.com/Asami-1/GDP.git
```
Not necessary, but you can also clone it using ssh. To that end, set up private/public ssh keys pairs and store the public key in your github account (see https://docs.github.com/fr/authentication/connecting-to-github-with-ssh/about-ssh) 

Once the repo has been cloned, switch to the branch related to your issue.

```console
git checkout  issue-12-Simulate_the_Env 
```

Carry out your work, when a functionality is finished commit it : 

```console
git commit . -m 'Added Dockerfile' 
```

* ```.``` means all the files in the current directory
* ```-m``` allows us to give a message to the commit

Note that you can add specific files, pass them as arguments instead of ```.```.


Once you have finished working and want to publish your work to the distant repository : 

```console
git push 
```

If you have finished your issue, you can start a new pull request from github. Go to https://github.com/Asami-1/GDP/compare and create it. **Make sure that there is no merge conflicts when you create the PR** and then assign a reviewer. Note that after pushing changes, the main page of the repo will suggest you to create a new PR.

## Reviewing somebody's work 

```console
git checkout colleague_branch 
git pull
```

Ensure that everything works, refer to the issue associated with the branch. You can add comments to the PR page on github if that's not the case. Otherwise validate the PR and merge the feature branch to the main branch.


IDEs such as VSCode also provide git integration, I recommend using that as they ease development. They also provide extensions to improve the usage of git. 



