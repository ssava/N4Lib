using System.Linq;

namespace N4Lib;

public sealed class Graph<TItem> where TItem : class
{
    public record Node(TItem Item);

    public record Edge(Node From, Node To, double Weight);

    private readonly HashSet<Edge> _edges = new();

    public IEnumerable<Edge> Edges
        => _edges;

    public IEnumerable<Node> Nodes()
    {
        IEnumerable<Node> startNodes = _edges.Select(e => e.From);
        IEnumerable<Node> destinationNodes = _edges.Select(x => x.To);

        return startNodes.Union(destinationNodes)
                         .ToHashSet();
    }

    public Edge AddEdge(Node from, Node to, double weight)
    {
        var edge = new Edge(from, to, weight);

        this._edges.Add(edge);

        return edge;
    }

    public bool ContainsNode(Node node)
        => _edges.Any(e => e.To == node || e.From == node);

    public IEnumerable<Node> NeighborsOf(Node node)
        => _edges.Where(e => e.From == node)
                 .Select(e => e.To)
                 .ToHashSet();
}