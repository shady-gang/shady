if [ "${ZSH_VERSION:-}" != "" ]; then
    SOURCE=${(%):-%N}
else
    SOURCE=${BASH_SOURCE[0]}
fi

CUR=`dirname $SOURCE`

export PATH="${CUR}/build/bin:${PATH:-}"
export LD_LIBRARY_PATH="${CUR}/build/lib:${LD_LIBRARY_PATH:-}"

if [ "${ZSH_VERSION:-}" != "" ]; then
    export fpath=(${CUR}/zsh ${fpath:-})
    compinit
fi
