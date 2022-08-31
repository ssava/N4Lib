namespace N4Lib;

public sealed class LossFunction
{
    /// <summary>
    /// Mean Squared Error (MSE for short)
    /// This loss function is generally used for regression tasks.
    /// </summary>
    /// <returns>MSE loss function instance</returns>
    public static readonly LossFunction MeanSquaredError
        = new((expected, actual) => {
            var squares = expected.Select((y, i) => Math.Pow(y - actual.ElementAt(i), 2));

            return (1/expected.Count()) * squares.Sum();
        });

    /// <summary>
    /// Binary Cross-Entropy (BCE for short).
    /// This loss function is used for binary classification.
    /// </summary>
    /// <returns>BCE loss function instance</returns>
    /// <remarks>This function need ONE output node and Sigmoid function</remarks>
    public static readonly LossFunction BinaryCrossEntropy
        = new((expected, actual) => {
            var correctedProbabilities = expected.Select(
                (y, i) => y != actual.ElementAt(i) ? 1 - y : y 
            );

            var logOfCorrectedProbabilities = correctedProbabilities.Select(p => Math.Log(p));

            return (-1/expected.Count()) * logOfCorrectedProbabilities.Sum();
        });

    /// <summary>
    /// Categorical Cross-Entropy (CCE for short).
    /// This loss function is used for multi/class classification.
    /// </summary>
    /// <returns>CCE loss function instance</returns>
    /// <remarks>This function need N output, where N is the number of classes and output node should use SoftMax function</remarks>
    // public static readonly LossFunction CategoricalCrossEntropy
    //     => new(x => x, x => x);

    /// <summary>
    /// Sparse Categorical Cross-Entropy (SCCE for short).
    /// This loss function is used for multi/class classification.
    /// Is similar to CCE but One-Hot Encode for output is not necessary
    /// </summary>
    /// <returns>CCE loss function instance</returns>
    /// <remarks>This function need N output, where N is the number of classes and output node should use SoftMax function</remarks>
    // public static readonly LossFunction SparseCategoricalCrossEntropy
    //     => new(x => x, x => x);

    private readonly Func<IEnumerable<double>, IEnumerable<double>, double> _function;

    private LossFunction(Func<IEnumerable<double>, IEnumerable<double>, double> loss)
        => this._function = loss;

    public double Loss(IEnumerable<double> actual, IEnumerable<double> expected)
    {
        if (actual.Count() != expected.Count())
            throw new InvalidOperationException("Both actual and expected vectors should have the same sizes!");

        return this._function(actual, expected);
    }
}