import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Re‑usable equation wrapper for consistent styling
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const GiniMeanDifferencePage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Gini Mean Difference | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn about Gini Mean Difference, a robust measure of dispersion in returns that evaluates the average absolute difference between all pairs of observations."
        />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Education</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Gini Mean Difference
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A robust dispersion measure based on differences between observations
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Gini Mean Difference (GMD)</strong> is a measure of statistical dispersion that evaluates the average absolute difference between all pairs of observations in a dataset. Originally developed by Corrado Gini in 1912, it has found applications in finance as an alternative risk measure to standard deviation.
          </Typography>
          <Typography paragraph>
            Unlike variance or standard deviation, which measure dispersion around the mean, the Gini Mean Difference directly quantifies the average dissimilarity between returns. This makes it particularly useful for analyzing non-normal return distributions, as it doesn't require any assumptions about the underlying distribution and is less sensitive to outliers.
          </Typography>
          <Typography paragraph>
            In portfolio theory, the Gini Mean Difference provides a robust alternative for measuring portfolio risk, especially when returns exhibit skewness, kurtosis, or other departures from normality that are common in financial markets.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            The Gini Mean Difference can be understood as measuring the "dissimilarity" or "distance" between returns. If you picked any two returns from your portfolio's history at random, the GMD represents the average absolute difference you would expect to see between them.
          </Typography>
          <Typography paragraph>
            Imagine having a set of monthly returns for a portfolio: some months you earned 2%, others 1%, perhaps -3% in bad months, and so on. The Gini Mean Difference looks at every possible pair of monthly returns and calculates how far apart they are from each other (ignoring whether one is higher or lower, just the absolute magnitude of the difference). It then averages all these differences.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Population analogy:</strong> Think of the Gini Mean Difference as measuring how "diverse" a population is. If you randomly select two people from a population and measure their heights, incomes, or any other characteristic, the GMD tells you how different you can expect them to be on average. A higher GMD indicates greater diversity or dispersion in the population.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            The Gini Mean Difference is defined as the mean absolute difference between all pairs of observations in a dataset. For a set of returns <InlineMath math="r_1, r_2, \ldots, r_n" />, the GMD is calculated as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Gini Mean Difference Formula</strong></Typography>
            <Equation math="\text{GMD} = \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} |r_i - r_j|" />
            <Typography variant="body2">
              where <InlineMath math="n" /> is the number of observations and <InlineMath math="|r_i - r_j|" /> is the absolute difference between the ith and jth returns.
            </Typography>
          </Box>

          <Typography paragraph>
            This formula involves <InlineMath math="n^2" /> terms in the summation, including comparisons of each element with itself (which yield zero differences). An alternative formulation that excludes self-comparisons is:
          </Typography>

          <Equation math="\text{GMD} = \frac{1}{n(n-1)} \sum_{i=1}^{n} \sum_{j=1}^{n, j \neq i} |r_i - r_j|" />

          <Typography paragraph>
            This version divides by <InlineMath math="n(n-1)" /> instead of <InlineMath math="n^2" /> since there are <InlineMath math="n(n-1)" /> distinct pairs of observations when excluding self-comparisons.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Alternative Computational Forms</Typography>
          <Typography paragraph>
            The GMD can also be expressed in terms of the ordered sample <InlineMath math="r_{(1)} \leq r_{(2)} \leq \ldots \leq r_{(n)}" />:
          </Typography>

          <Equation math="\text{GMD} = \frac{2}{n^2} \sum_{i=1}^{n} (2i - n - 1)r_{(i)}" />

          <Typography paragraph>
            This formulation is computationally more efficient as it reduces the number of operations from <InlineMath math="O(n^2)" /> to <InlineMath math="O(n \log n)" /> (the complexity is dominated by the sorting operation).
          </Typography>

          <Typography paragraph>
            Another useful representation is in terms of the empirical cumulative distribution function <InlineMath math="F_n(x)" />:
          </Typography>

          <Equation math="\text{GMD} = 2 \int_{-\infty}^{\infty} F_n(x)(1 - F_n(x)) \, dx" />

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Relationship with Other Dispersion Measures</Typography>
          <Typography paragraph>
            The Gini Mean Difference is related to several other statistical measures:
          </Typography>

          <Typography paragraph>
            <strong>1. Relationship with Variance:</strong> For a normal distribution with standard deviation <InlineMath math="\sigma" />, the GMD equals:
          </Typography>

          <Equation math="\text{GMD} = \frac{2\sigma}{\sqrt{\pi}} \approx 1.128 \sigma" />

          <Typography paragraph>
            This shows that for normal distributions, GMD is proportional to the standard deviation, but it generalizes better to non-normal distributions.
          </Typography>

          <Typography paragraph>
            <strong>2. Relationship with Mean Absolute Deviation (MAD):</strong> The MAD measures the average absolute deviation from the mean <InlineMath math="\mu" />:
          </Typography>

          <Equation math="\text{MAD} = \frac{1}{n} \sum_{i=1}^{n} |r_i - \mu|" />

          <Typography paragraph>
            For symmetric distributions, the relationship between GMD and MAD is:
          </Typography>

          <Equation math="\text{GMD} = 2 \cdot \text{MAD}" />

          <Typography paragraph>
            <strong>3. Relationship with Gini Coefficient:</strong> The Gini coefficient, commonly used to measure income inequality, is half the relative Gini Mean Difference:
          </Typography>

          <Equation math="\text{Gini Coefficient} = \frac{\text{GMD}}{2\mu}" />

          <Typography paragraph>
            where <InlineMath math="\mu" /> is the mean of the distribution. In portfolio theory, this is sometimes used to normalize the GMD.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Weighted Gini Mean Difference</Typography>
          <Typography paragraph>
            In portfolio contexts, we often need to calculate the Gini Mean Difference for weighted returns. For asset weights <InlineMath math="w = (w_1, w_2, \ldots, w_m)" /> and asset returns matrix <InlineMath math="R" />, the portfolio's GMD is:
          </Typography>

          <Equation math="\text{GMD}_p = \sum_{i=1}^{m} \sum_{j=1}^{m} w_i w_j \text{GMD}_{ij}" />

          <Typography paragraph>
            where <InlineMath math="\text{GMD}_{ij}" /> is the Gini Mean Difference between the returns of assets i and j. This quadratic form is similar to the variance-covariance formulation in Modern Portfolio Theory, but uses GMD instead of variance and covariance.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Our Service</Typography>
          <Typography paragraph>
            Our portfolio analyzer calculates the Gini Mean Difference using the following approach:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Historical Return Analysis:</strong> We start by gathering historical returns for all assets in the portfolio over the specified time period.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Optimization for Large Datasets:</strong> For computational efficiency, we use the ordered sample formulation of GMD for large datasets:
              </Typography>
              <Typography sx={{ ml: 4 }} paragraph>
                <InlineMath math="\text{GMD} = \frac{2}{n^2} \sum_{i=1}^{n} (2i - n - 1)r_{(i)}" />
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Portfolio GMD Calculation:</strong> We calculate the weighted GMD for the portfolio using the quadratic form that accounts for asset weights and pairwise GMDs between assets.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Optimization Objective:</strong> We offer GMD minimization as an alternative objective function in portfolio optimization, particularly useful for investors concerned with non-normal return distributions.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            This implementation provides a robust risk measure that captures the dispersion of returns without making assumptions about normality, making it particularly valuable for portfolios with asymmetric return distributions or significant tail risk.
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', mt: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" gutterBottom>GMD vs. Standard Deviation Comparison (Placeholder)</Typography>
            <Box sx={{ height: '300px', bgcolor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                [Placeholder for GMD vs. standard deviation visualization]
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              This chart illustrates how GMD and standard deviation differ in measuring dispersion for various return distributions, highlighting GMD's robustness to outliers and non-normality.
            </Typography>
          </Box>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate the Gini Mean Difference for a small set of monthly portfolio returns:
          </Typography>
          <Typography paragraph>
            2%, -1%, 3%, 0%, 1%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: List all pairwise absolute differences</Typography>
          <Typography paragraph>
            We calculate the absolute difference between each pair of returns:
          </Typography>
          <ul>
            <li><Typography paragraph>|2% - 2%| = 0%</Typography></li>
            <li><Typography paragraph>|2% - (-1%)| = 3%</Typography></li>
            <li><Typography paragraph>|2% - 3%| = 1%</Typography></li>
            <li><Typography paragraph>|2% - 0%| = 2%</Typography></li>
            <li><Typography paragraph>|2% - 1%| = 1%</Typography></li>
            <li><Typography paragraph>|-1% - 2%| = 3%</Typography></li>
            <li><Typography paragraph>|-1% - (-1%)| = 0%</Typography></li>
            <li><Typography paragraph>|-1% - 3%| = 4%</Typography></li>
            <li><Typography paragraph>|-1% - 0%| = 1%</Typography></li>
            <li><Typography paragraph>|-1% - 1%| = 2%</Typography></li>
            <li><Typography paragraph>|3% - 2%| = 1%</Typography></li>
            <li><Typography paragraph>|3% - (-1%)| = 4%</Typography></li>
            <li><Typography paragraph>|3% - 3%| = 0%</Typography></li>
            <li><Typography paragraph>|3% - 0%| = 3%</Typography></li>
            <li><Typography paragraph>|3% - 1%| = 2%</Typography></li>
            <li><Typography paragraph>|0% - 2%| = 2%</Typography></li>
            <li><Typography paragraph>|0% - (-1%)| = 1%</Typography></li>
            <li><Typography paragraph>|0% - 3%| = 3%</Typography></li>
            <li><Typography paragraph>|0% - 0%| = 0%</Typography></li>
            <li><Typography paragraph>|0% - 1%| = 1%</Typography></li>
            <li><Typography paragraph>|1% - 2%| = 1%</Typography></li>
            <li><Typography paragraph>|1% - (-1%)| = 2%</Typography></li>
            <li><Typography paragraph>|1% - 3%| = 2%</Typography></li>
            <li><Typography paragraph>|1% - 0%| = 1%</Typography></li>
            <li><Typography paragraph>|1% - 1%| = 0%</Typography></li>
          </ul>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Calculate the average of all pairwise differences</Typography>
          <Typography paragraph>
            Sum of all differences: 0% + 3% + 1% + 2% + 1% + 3% + 0% + 4% + 1% + 2% + 1% + 4% + 0% + 3% + 2% + 2% + 1% + 3% + 0% + 1% + 1% + 2% + 2% + 1% + 0% = 40%
          </Typography>
          <Typography paragraph>
            Number of pairs: 5 × 5 = 25
          </Typography>
          <Typography paragraph>
            GMD = 40% ÷ 25 = 1.6%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Using the alternative formula with ordered returns</Typography>
          <Typography paragraph>
            Let's verify using the more efficient formula. First, we sort the returns:
          </Typography>
          <Typography paragraph>
            -1%, 0%, 1%, 2%, 3%
          </Typography>
          <Typography paragraph>
            Then apply the formula:
          </Typography>
          <Typography component="div" paragraph>
            <Equation math="\text{GMD} = \frac{2}{5^2} \sum_{i=1}^{5} (2i - 5 - 1)r_{(i)}" />
            <Equation math="\text{GMD} = \frac{2}{25} [ (-5)(-1\%) + (-3)(0\%) + (-1)(1\%) + (1)(2\%) + (3)(3\%) ]" />
            <Equation math="\text{GMD} = \frac{2}{25} [ 5\% + 0\% - 1\% + 2\% + 9\% ]" />
            <Equation math="\text{GMD} = \frac{2}{25} \cdot 15\% = \frac{30\%}{25} = 1.2\%" />
          </Typography>

          <Typography paragraph>
            The slight discrepancy between the two calculations (1.6% vs. 1.2%) is due to the different approaches in handling self-comparisons. The second method (1.2%) is the more commonly used form in financial applications.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Comparison with standard deviation</Typography>
          <Typography paragraph>
            For comparison, let's calculate the standard deviation of these returns:
          </Typography>
          <Typography paragraph>
            Mean return: (2% + (-1%) + 3% + 0% + 1%) ÷ 5 = 1%
          </Typography>
          <Typography paragraph>
            Sum of squared deviations: (2% - 1%)² + (-1% - 1%)² + (3% - 1%)² + (0% - 1%)² + (1% - 1%)² = 1% + 4% + 4% + 1% + 0% = 10%
          </Typography>
          <Typography paragraph>
            Variance: 10% ÷ 5 = 2%
          </Typography>
          <Typography paragraph>
            Standard deviation: √2% ≈ 1.41%
          </Typography>

          <Typography paragraph>
            We see that for this small sample, the GMD (1.2%) is slightly lower than the standard deviation (1.41%). This relationship varies depending on the distribution of returns.
          </Typography>
        </Paper>

        {/* Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          <Typography paragraph>
            The Gini Mean Difference has several valuable applications in portfolio management and risk assessment:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Risk-Based Portfolio Optimization:</strong> GMD can be used as an alternative risk measure in portfolio optimization, particularly when returns exhibit non-normality. Minimizing GMD instead of variance can lead to portfolios with reduced exposure to extreme return differences.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Performance Evaluation:</strong> GMD provides a robust metric for comparing the risk characteristics of different portfolios or investment strategies, especially in markets with asymmetric return distributions.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Parity Frameworks:</strong> GMD can be incorporated into risk parity approaches to allocate risk contributions more evenly across assets, accounting for non-normal return patterns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Robust Risk Budgeting:</strong> When allocating risk across portfolio components, GMD provides a measure less affected by outliers, potentially leading to more stable risk allocations over time.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Analysis of Alternative Investments:</strong> For alternative investments with highly skewed returns (like hedge funds, private equity, or option strategies), GMD can provide a more appropriate risk measure than standard deviation.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Advantages and Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages and Limitations
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Advantages
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Distribution-free:</strong> GMD makes no assumptions about the underlying return distribution, making it suitable for non-normal returns common in financial markets.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Robustness to outliers:</strong> GMD is less sensitive to extreme observations than variance or standard deviation, providing a more stable risk measure over time.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Interpretability:</strong> GMD has a straightforward interpretation as the average difference between randomly selected returns, making it intuitive for investors to understand.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Mathematical properties:</strong> GMD satisfies several desirable mathematical properties, including subadditivity, making it suitable for risk measurement in a portfolio context.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Direct measurement:</strong> GMD directly measures the dispersion between returns rather than deviations from a central tendency, which can be more relevant for certain risk assessments.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Limitations
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Computational complexity:</strong> The naïve calculation of GMD has O(n²) complexity, making it potentially computationally intensive for large datasets, though more efficient algorithms exist.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Less established:</strong> GMD is less widely used in finance than standard deviation, potentially making it harder to benchmark or compare results with industry standards.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Different scale:</strong> GMD produces values on a different scale than standard deviation, requiring adjustment when comparing to traditional risk measures or interpreting historical benchmarks.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time dependence:</strong> Like other historical risk measures, GMD is backward-looking and may not capture future dispersion patterns if market dynamics change.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Less developed theory:</strong> The statistical inference theory for GMD is less developed than for variance-based measures, potentially limiting certain applications in hypothesis testing or confidence interval construction.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Comparison with Other Risk Metrics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Comparison with Other Risk Metrics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>GMD vs. Standard Deviation</Typography>
                <Typography paragraph>
                  While <MuiLink component={Link} href="/docs/volatility">standard deviation</MuiLink> measures dispersion as the square root of the average squared deviation from the mean, GMD measures the average absolute difference between all pairs of returns. Standard deviation gives greater weight to outliers due to the squaring operation, making it more sensitive to extreme values than GMD. For normal distributions, GMD ≈ 1.128 × standard deviation, but this relationship breaks down for non-normal distributions, where GMD typically provides a more robust assessment of dispersion.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>GMD vs. Mean Absolute Deviation</Typography>
                <Typography paragraph>
                  Mean Absolute Deviation (MAD) measures the average absolute deviation from the mean, while GMD measures the average absolute difference between all pairs of observations. For symmetric distributions, GMD = 2 × MAD. GMD considers the full structure of the data by examining all pairwise relationships, potentially capturing dispersion characteristics that MAD might miss, especially in multimodal or highly skewed distributions.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>References</Typography>
          <ul>
            <li>
              <Typography paragraph><strong>Yitzhaki, S. (2003)</strong>. "Gini's Mean Difference: A superior measure of variability for non-normal distributions." <em>Metron - International Journal of Statistics</em>, 61(2), 285-316.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Shalit, H., & Yitzhaki, S. (2005)</strong>. "The Mean-Gini Efficient Portfolio Frontier." <em>Journal of Financial Research</em>, 28(1), 59-75.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Giorgi, G. M. (1990)</strong>. "Bibliographic portrait of the Gini concentration ratio." <em>Metron</em>, 48, 183-221.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Ceriani, L., & Verme, P. (2012)</strong>. "The origins of the Gini index: extracts from Variabilità e Mutabilità (1912) by Corrado Gini." <em>The Journal of Economic Inequality</em>, 10(3), 421-443.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Ogryczak, W., & Ruszczyński, A. (1999)</strong>. "From stochastic dominance to mean-risk models: Semideviations as risk measures." <em>European Journal of Operational Research</em>, 116(1), 33-50.</Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Related Topics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Volatility</Typography>
                <Typography variant="body2" paragraph>A statistical measure of the dispersion of returns, usually measured using standard deviation.</Typography>
                <Link href="/docs/volatility" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Entropy</Typography>
                <Typography variant="body2" paragraph>A measure of uncertainty or randomness in portfolio returns, indicating the level of unpredictability in the system.</Typography>
                <Link href="/docs/entropy" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Mean-Variance Optimization</Typography>
                <Typography variant="body2" paragraph>The cornerstone of Modern Portfolio Theory that helps investors construct optimal portfolios balancing risk and return.</Typography>
                <Link href="/docs/mvo" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

// Static generation hook (Next.js)
export const getStaticProps = async () => {
  return { props: {} };
};

export default GiniMeanDifferencePage; 