using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;


public class Seeker : MonoBehaviour
{
    public State_Machine stateMachine;
    public NavMeshAgent agent;
    public NavMeshAgent Agent { get => agent; }
    public SeekPath path;

    [SerializeField]
    private string currentstate;
    public Vector3 lastseenposition;
    private GameObject player;
    public float fieldOfview = 60f;
    public float fieldOfviewDistance = 20f;
    private void Start()
    {
        stateMachine = GetComponent<State_Machine>();
        agent = GetComponent<NavMeshAgent>();
        player = GameObject.FindGameObjectsWithTag("Player")[0];
        stateMachine.Initialise();
    }

    public void Reset(Vector3 position){ 
        transform.position = position;
        stateMachine.Initialise();
    }


    public bool canSee()
    {

        if (player != null)
        {
            if (Vector3.Distance(player.transform.position, transform.position) < fieldOfviewDistance)
            {
                Vector3 targetdirection = player.transform.position - transform.position;
                float angletoplayer = Vector3.Angle(targetdirection, transform.forward);
                if (angletoplayer <= fieldOfview && angletoplayer >= -fieldOfview)
                {
                    Ray ray = new Ray(transform.position, targetdirection);
                    RaycastHit hitinfo = new RaycastHit();
                    if (Physics.Raycast(ray, out hitinfo, fieldOfviewDistance))
                    {
                        if (hitinfo.transform.gameObject == player)
                        {
                            lastseenposition = player.transform.position;
                            return true;
                        }

                    }
                }
            }
        }

        return false;



    }

    //private void OnCollisionEnter(Collision collision)
    //{
    //    if (collision.gameObject.CompareTag("Player"))
    //    {  
    //        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    //    }
    //}


}