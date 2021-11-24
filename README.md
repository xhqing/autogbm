# autogbm
## Development Environment
Using Ubuntu OS and Python3.6 with docker, you can generate your own container as the following:
```
git clone git@github.com:xhqing/autogbm.git
cd autogbm/dev_env
docker build autogbm-dev .
cd ..
docker run -it -v "$(pwd):/app/workdir/autogbm" --name=autogbm-dev autogbm-dev
```
and then (within the container), change dir to autogbm `cd autogbm`, then run `pipenvPython` you will get a virtualenv for Python3.6, then using `pipenvInstall --dev` to install all packages list in Pipfile (including dev-packages), and then you can run `pipenv run python example.py` for a test, or using the pipenv subshell by exec `pipenv shell` and then exec `python example.py` within the subshell for a test, the order of executing commands as follows:
```
# confirm you are in /app/workdir
pwd

# change dir
cd autogbm

# generate python3.6 virtual environment
pipenvPython 

# install all packages needed (including dev-packages) from Pipfile
pipenvInstall --dev

# test example.py
pipenv run python example.py
```
see more about [pipenv website](https://pipenv.pypa.io/en/latest/), and check the bottom of `~.zshrc`file to know about my pipenv alias.
## Reference Papers
Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin ["CatBoost: gradient boosting with categorical features support"](http://learningsys.org/nips17/assets/papers/paper_11.pdf). Workshop on ML Systems at NIPS 2017.

Tianqi Chen and Carlos Guestrin. [XGBoost: A Scalable Tree Boosting System](http://arxiv.org/abs/1603.02754). In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016


## License
This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/xhqing/autogbm/blob/master/LICENSE) for additional details.
