
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.AI;

public class Hider : Agent
{
    public Seeker seeker;
    public float moveSpeed = 2f;
    public float turnSpeed = 180f;
    public Rigidbody rigidbody;

    public override void OnEpisodeBegin()
    {
        rigidbody.linearVelocity = Vector3.zero;
        rigidbody.angularVelocity = Vector3.zero;

        transform.position = GetRandomPoint(Vector3.zero, 7f);
        seeker.Reset(GetRandomPoint(Vector3.zero, 7f));

        transform.LookAt(seeker.transform);
        seeker.transform.LookAt(transform);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int moveAction = actions.DiscreteActions[0];
        Vector3 moveDir = Vector3.zero;
        float rotation = 0f;

        if (moveAction == 1) moveDir = transform.forward;
        else if (moveAction == 2) moveDir = -transform.forward;
        else if (moveAction == 3) rotation = -1f;
        else if (moveAction == 4) rotation = 1f;

        rigidbody.MovePosition(transform.position + moveDir * moveSpeed * Time.fixedDeltaTime);
        transform.Rotate(Vector3.up * rotation * turnSpeed * Time.fixedDeltaTime);

        AddReward(0.001f);
        if (transform.position.y < -1){
            EndEpisode();
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Seeker"))
        {
            SetReward(-1f);
            EndEpisode();
        }
    }

    public static Vector3 GetRandomPoint(Vector3 center, float maxDistance)
    {
        Vector3 randomPos = Random.insideUnitSphere * maxDistance + center;
        NavMeshHit hit;
        NavMesh.SamplePosition(randomPos, out hit, maxDistance, NavMesh.AllAreas);
        return hit.position;
    }
}