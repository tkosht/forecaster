#!/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

nm=$(basename $(pwd))

git init
git remote add origin git@github.com:tkosht/$nm.git
git fetch
git pull origin master

git add .
git commit -m 'init: initial commit'
git push --set-upstream origin master
