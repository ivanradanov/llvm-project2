
FAIL=0

if [ "$1" == "" ]; then
	echo Please provide arg 1>&2
	FAIL=1
fi

if ! readlink -f "$1" > /dev/null ; then
	echo Warning: Path to enable "'$1'" does not exist 1>&2
	FAIL=1
fi

if [ "$FAIL" == "0" ]; then
	PATH="$(readlink -f "$1")/bin/:$PATH"
	LD_LIBRARY_PATH="$(readlink -f "$1")/lib/:$LD_LIBRARY_PATH"
	LIBRARY_PATH="$(readlink -f "$1")/lib/:$LIBRARY_PATH"
fi

return $FAIL
