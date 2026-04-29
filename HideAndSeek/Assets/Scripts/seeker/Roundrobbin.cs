using NUnit.Framework;
using UnityEditorInternal;
using UnityEngine;
using System.Collections.Generic;


public class Roundrobbin : BaseState
{
    public int waypointindex = -1;

    public override void Perform()
    {
        Logic();
    }

    
    public void Logic()
    {
        if (!seeker.canSee()) {
            if (seeker.Agent.remainingDistance < 0.2f)
            {
                if (waypointindex < seeker.path.waypoints.Count - 1)
                {
                    waypointindex++;
                }
                else
                {
                    waypointindex = 0;
                }
                seeker.Agent.SetDestination(seeker.path.waypoints[waypointindex].position);
            }
        }
        else
        {
            seeker.stateMachine.ChangeState(seeker.stateMachine.seehider);
        }

    }
}
