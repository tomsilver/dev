# shut up about zsh
export BASH_SILENCE_DEPRECATION_WARNING=1

# make python hashing deterministic
export PYTHONHASHSEED=0

# fast downward
export FD_EXEC_PATH="/Users/tom/downward"

# fast forward
export FF_PATH="/Users/tom/phd/FF/ff"

# rbenv (ruby)
eval "$(rbenv init - bash)"

# cd and ls
function cs {
    builtin cd "$@" && ls -F
}

# profiling with python / snakeviz
profile(){
    python -m cProfile -o /tmp/program.prof "$@" && snakeviz /tmp/program.prof
}

# find and replace for git
git-replace(){
    git grep -l "$1" | xargs sed -i '' -e "s/$1/$2/g"
}

# list branches by most recent modification
git-branch(){
    git branch -v --sort='-authordate:iso8601' --format='%(align:width=50)%(refname:short)%(end)%(authordate:relative)'
}

# switch to the main branch and delete the old branch
git-forward() {
  # Get the current branch name
  current_branch=$(git branch --show-current)

  # Check for unstaged changes
  if git diff-index --quiet HEAD --; then

    # Switch to the main branch
    git checkout main
    
    # Delete the previous branch
    git branch -D "$current_branch"

    # Pull changes
    git pull
  else
    echo "Unstaged changes present in '$current_branch'. Please commit or stash changes before deleting the branch."
  fi
}

# the typical thing
git-gg() {
    git add -u
    git commit -m wip
    git push origin HEAD
}

# no verify
git-gg-nv() {
    git add -u
    git commit -m wip --no-verify
    git push origin HEAD
}

# remove unused imports
remove-imports(){
    autoflake --in-place --remove-all-unused-imports "$@"
}

# Start the SSH agent if necessary and add all ssh keys
SSH_ENV="$HOME/.ssh/agent-environment"

function start_agent {
    echo "Initializing new SSH agent..."
    /usr/bin/ssh-agent | sed 's/^echo/#echo/' > "${SSH_ENV}"
    echo succeeded
    chmod 600 "${SSH_ENV}"
    . "${SSH_ENV}" > /dev/null
    # Find all private SSH keys in ~/.ssh and add them
    find ~/.ssh/ -type f -exec grep -l "PRIVATE" {} \; | xargs ssh-add &> /dev/null
}

# Source SSH settings, if applicable
if [ -f "${SSH_ENV}" ]; then
    . "${SSH_ENV}" > /dev/null
    ps -ef | grep ${SSH_AGENT_PID} | grep ssh-agent$ > /dev/null || {
        start_agent;
    }
else
    start_agent;
fi

function venv {
    if [ -d .venv ]; then
        source .venv/bin/activate
    elif [ -d venv ]; then
        source venv/bin/activate
    else
        uv venv --python=3.11
        source .venv/bin/activate
    fi
}
