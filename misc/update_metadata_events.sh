#!/usr/bin/env sh

# Get start/end folder ids
if (([ -z "$1" ] || [-z "$2"] || [-z "$3"]))
  then
    echo "Usage: sh postprocess_sim_pokes.sh rootfol folstart folend"
	 return;
fi

rootfol=$1
for id in $(seq $2 $3);
do
	echo "Dataset: $rootfol/poke$id (End: poke$3)"
	cmd="devel/lib/baxter_motion_simulator/update_metadata_events --pokefolder $rootfol/poke$id --modelfolder src/learn_physics_models/models/household_objects/"
	eval $cmd
done
