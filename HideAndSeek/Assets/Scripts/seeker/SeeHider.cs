using JetBrains.Annotations;
using UnityEngine;

public class SeeHider : BaseState
{
    public override void Perform()
    {
        Logic();
    }
    public void Logic() {
        if (seeker.canSee())
        {
            seeker.Agent.SetDestination(seeker.lastseenposition);
        }
        else {
            if (seeker.Agent.remainingDistance < 0.2f) {
                seeker.stateMachine.ChangeState(seeker.stateMachine.roundrobbin);
            }
        }
    
    }
}
