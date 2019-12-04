# Initialize a directory as a github repo.

SET readme=README.md
SET ign=.gitignore

if not exist %ign% (
	echo git_init.bat >> %ign%
	echo submit.bat >> %ign%
)

if not exist %readme% (
	echo # README\n >> %readme%
) else (
	git init
	git add .
	git commit -m "Initialization"
	git remote add origin %1
	git push -u origin master
)

echo Done.
pause
