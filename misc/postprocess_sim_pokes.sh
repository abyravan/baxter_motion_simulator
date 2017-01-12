#!/usr/bin/env sh

# Get start/end folder ids
if (([ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]))
  then
    echo "Usage: sh postprocess_sim_pokes.sh rootfol folstart folend step"
	 return;
fi

rootfol=$1
start=$2
end=$3
step=$4
for id in $(seq $start $step $end);
do
	echo "ID: [$id/$end]"
	cmd="devel/lib/baxter_motion_simulator/postprocess_simulated_pokes --pokeroot $rootfol --startid $id --endid $((id+step)) --modelfolder src/learn_physics_models/models/household_objects/ --steplist [1,2,3,4,5,6,7,8,9]"
	eval $cmd
done
