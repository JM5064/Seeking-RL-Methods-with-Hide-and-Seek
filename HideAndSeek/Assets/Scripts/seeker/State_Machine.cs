using UnityEngine;

public class State_Machine : MonoBehaviour
{
    public BaseState activestate;
    public Roundrobbin roundrobbin;
    public SeeHider seehider;
    public void Initialise() { 
        roundrobbin = new Roundrobbin();
        seehider = new SeeHider();
        ChangeState(roundrobbin);

    } 

    // Update is called once per frame
    void Update()
    {
        if (activestate != null) activestate.Perform();
        
    }
    public void ChangeState(BaseState newstate) {

        activestate = newstate;
        if (activestate != null) { 
            activestate.machine = this;
            activestate.seeker = GetComponent<Seeker>();
            activestate.Perform();
        }
    
    }


}
