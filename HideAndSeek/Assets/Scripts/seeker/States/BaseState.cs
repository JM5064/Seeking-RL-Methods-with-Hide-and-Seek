using UnityEngine;

public abstract class BaseState 
{
    public Seeker seeker;
    public State_Machine machine;
    // Update is called once per frame
    public abstract void Perform();

}
