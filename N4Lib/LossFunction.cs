namespace N4Lib;

public sealed class LossFunction
{
    /// <summary>
    /// Mean Squared Error (MSE for short)
    /// This loss function is generally used for regression tasks.
    /// </summary>
    /// <returns>MSE loss function instance</returns>
    public static readonly LossFunction MeanSquaredError
        = new(x => x * 2, x => 2);

    /// <summary>
    /// Binary Cross-Entropy (BCE for short).
    /// This loss function is used for binary classification.
    /// </summary>
    /// <returns>BCE loss function instance</returns>
    /// <remarks>This function need ONE output node and Sigmoid function</remarks>
    public static readonly LossFunction BinaryCrossEntropy
        => new(x => x, x => x);

    /// <summary>
    /// Categorical Cross-Entropy (CCE for short).
    /// This loss function is used for multi/class classification.
    /// </summary>
    /// <returns>CCE loss function instance</returns>
    /// <remarks>This function need N output, where N is the number of classes and output node should use SoftMax function</remarks>
    public static readonly LossFunction CategoricalCrossEntropy
        => new(x => x, x => x);

    /// <summary>
    /// Sparse Categorical Cross-Entropy (SCCE for short).
    /// This loss function is used for multi/class classification.
    /// Is similar to CCE but One-Hot Encode for output is not necessary
    /// </summary>
    /// <returns>CCE loss function instance</returns>
    /// <remarks>This function need N output, where N is the number of classes and output node should use SoftMax function</remarks>
    public static readonly LossFunction SparseCategoricalCrossEntropy
        => new(x => x, x => x);

    private readonly Func<double, double> _function;

    private readonly Func<double, double> _prime;

    private LossFunction(Func<double, double> loss, Func<double, double> prime)
        => (this._function, this._prime) = (loss, prime);

    public double Loss(double x)
        => this._function(x);

    public double Prime(double x)
        => this._prime(x);
}