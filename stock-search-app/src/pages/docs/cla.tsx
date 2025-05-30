import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink,
  Divider
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Reusable Equation component for consistent math rendering
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const CriticalLineAlgorithmPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Critical Line Algorithm for Indian Markets | QuantPort India Docs</title>
        <meta name="description" content="Master the Critical Line Algorithm (CLA) for optimizing Indian stock portfolios. Learn Markowitz's exact approach for tracing efficient frontiers of NSE/BSE securities." />
        <meta property="og:title" content="Critical Line Algorithm for Indian Markets | QuantPort India Docs" />
        <meta property="og:description" content="Master the Critical Line Algorithm (CLA) for optimizing Indian stock portfolios. Learn Markowitz's exact approach for tracing efficient frontiers of NSE/BSE securities." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/cla" />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Docs
            </Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">
              ← Back to Portfolio Optimizer
            </Button>
          </Link>
        </Box>
        
        {/* Title Section */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Critical Line Algorithm (CLA)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A foundational method for tracing the complete efficient frontier in portfolio optimization
          </Typography>
        </Box>
        
        {/* Overview Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            The <strong>Critical Line Algorithm (CLA)</strong> is a classic and foundational method developed by Nobel laureate 
            Harry Markowitz. It directly solves the constrained quadratic optimization problem encountered in portfolio optimization. 
            CLA provides a systematic and mathematically precise approach to tracing the full efficient frontier—representing 
            optimal portfolios that offer the best possible return for a given level of risk or the lowest risk for a 
            specified return.
          </Typography>
          <Typography paragraph>
            Unlike simpler numerical methods, CLA explicitly identifies critical points (portfolio transitions) where constraints 
            change, offering detailed insights into how portfolios shift from one optimal solution to another as constraints vary. 
            This makes it particularly valuable for understanding the structural changes in portfolio composition across the 
            efficient frontier.
          </Typography>
          <Typography paragraph>
            The CLA is often considered the "gold standard" for portfolio optimization problems with linear equality and inequality 
            constraints, as it provides an exact, analytical solution rather than relying on numerical approximations that may 
            converge to suboptimal results.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Imagine you're planning a road trip and want to find the fastest route with the least fuel usage. However, you must 
            also consider factors like traffic, road conditions, and tolls, which change dynamically along your journey. The 
            Critical Line Algorithm is akin to meticulously mapping out every potential turning point along your route where 
            you'd reconsider your optimal path due to changes in conditions.
          </Typography>
          <Typography paragraph>
            Similarly, in portfolio optimization, CLA identifies all these crucial turning points ("critical lines") where 
            portfolio composition shifts, thoroughly outlining how optimal allocations change under different scenarios or constraints.
          </Typography>
          <Typography paragraph>
            In practical terms, as you move along the efficient frontier from high-risk, high-return portfolios to low-risk, 
            low-return portfolios, the weight of each asset doesn't change linearly. Instead, assets enter and leave the optimal 
            portfolio at specific points. The CLA precisely identifies these transition points, allowing investors to understand 
            exactly how portfolio structure evolves across the risk-return spectrum.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Example:</strong> Consider a simple portfolio of three stocks: A, B, and C. At high expected returns, the 
              optimal portfolio might include only stocks A and B. As you accept lower returns for reduced risk, at some precise 
              point, stock C enters the optimal portfolio while the allocation to stock A decreases. The CLA identifies exactly 
              where this transition occurs and how the weights should be adjusted, continuing this process until the minimum 
              risk portfolio is reached.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical Explanation
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            General Optimization Problem
          </Typography>
          <Typography paragraph>
            The CLA solves the following constrained quadratic optimization problem:
          </Typography>
          <Equation math="\begin{aligned} & \underset{w}{\text{minimize}} & & w^\top \Sigma w \\ & \text{subject to} & & w^\top \mathbf{1} = 1 \\ & & & w^\top \mu = \mu^* \\ & & & w_i \geq 0, \quad \forall i \end{aligned}" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="w" /> is the portfolio weight vector.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\Sigma" /> is the covariance matrix of asset returns.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\mu" /> is the expected returns vector.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\mu^*" /> is the desired target portfolio return.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\mathbf{1}" /> is a vector of ones.
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            The Critical Line Concept
          </Typography>
          <Typography paragraph>
            The key insight of Markowitz's Critical Line Algorithm is that the efficient frontier consists of connected line 
            segments in the weight space. Each line segment is determined by a specific set of active constraints (assets at 
            their lower bounds, typically zero for no-short-selling constraints). As you move along the efficient frontier, 
            the set of active constraints changes at specific points called "corner portfolios" or "turning points."
          </Typography>
          
          <Typography paragraph>
            Mathematically, this is represented using the Karush-Kuhn-Tucker (KKT) conditions. For the standard form of the problem:
          </Typography>
          
          <Equation math="\begin{aligned} \mathcal{L}(w, \lambda_1, \lambda_2, \mu) = w^\top \Sigma w - \lambda_1 (w^\top \mathbf{1} - 1) - \lambda_2 (w^\top \mu - \mu^*) - \sum_{i} \gamma_i w_i \end{aligned}" />
          
          <Typography paragraph>
            where <InlineMath math="\lambda_1, \lambda_2" /> are Lagrange multipliers for equality constraints and 
            <InlineMath math="\gamma_i" /> are KKT multipliers for inequality constraints.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Algorithm Steps
          </Typography>
          <Typography paragraph>
            The CLA proceeds through the following steps:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Initialization</strong>: Start with the highest attainable expected return portfolio (typically invested 
                entirely in the highest-return asset).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculate Direction</strong>: Determine the direction in weight space that maintains all constraints while 
                reducing variance maximally.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Identify Next Turning Point</strong>: Calculate how far to move in this direction until a new constraint 
                becomes active (an asset enters or leaves the portfolio).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Update Active Set</strong>: Update the set of active constraints and recalculate the direction for the 
                next segment.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Repeat</strong>: Continue this process until the minimum variance portfolio is reached.
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            This procedure traces out the entire efficient frontier segment by segment, identifying each critical turning point 
            where the portfolio structure changes.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Analytical Solutions
          </Typography>
          <Typography paragraph>
            For each segment of the critical line, with a given set of active inequality constraints, the optimal weights can be 
            expressed as a function of the target return <InlineMath math="\mu^*" />:
          </Typography>
          
          <Equation math="w = \gamma + \delta\mu^*" />
          
          <Typography paragraph>
            where <InlineMath math="\gamma" /> and <InlineMath math="\delta" /> are constant vectors for each segment, derived from 
            the covariance matrix, expected returns, and the set of active constraints. This linear relationship between weights and 
            target return within each segment is what gives the method its name—each segment is a "critical line" in the weight space.
          </Typography>
        </Paper>
        
        {/* Variants Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Variants in Implementation
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            1. Mean-Variance Optimization (MVO) variant of CLA
          </Typography>
          <Typography paragraph>
            This method maximizes the Sharpe Ratio, balancing expected return against volatility relative to a risk-free rate 
            (<InlineMath math="R_f" />). Mathematically, it solves:
          </Typography>
          
          <Equation math="\begin{aligned} & \underset{w}{\text{maximize}} & & \frac{w^\top \mu - R_f}{\sqrt{w^\top \Sigma w}} \\ & \text{subject to} & & w^\top \mathbf{1} = 1 \\ & & & w_i \geq 0 \end{aligned}" />
          
          <Typography paragraph>
            This yields the tangency portfolio with the highest possible Sharpe Ratio. The tangency portfolio represents the 
            point where a line from the risk-free rate touches the efficient frontier, offering the greatest excess return per 
            unit of risk.
          </Typography>
          <Typography paragraph>
            In the context of CLA, this specific portfolio is identified by systematically tracing the efficient frontier and 
            selecting the portfolio with maximum Sharpe ratio, which represents a specific point among the critical points 
            identified by the algorithm.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            2. Minimum Volatility (MinVol) variant of CLA
          </Typography>
          <Typography paragraph>
            This variant purely focuses on minimizing volatility, disregarding explicit return targets:
          </Typography>
          
          <Equation math="\begin{aligned} & \underset{w}{\text{minimize}} & & w^\top \Sigma w \\ & \text{subject to} & & w^\top \mathbf{1} = 1 \\ & & & w_i \geq 0 \end{aligned}" />
          
          <Typography paragraph>
            This produces the least risky portfolio possible given the constraints. The minimum volatility portfolio 
            represents the leftmost point on the efficient frontier—the portfolio with the absolute lowest risk among all 
            feasible portfolios.
          </Typography>
          <Typography paragraph>
            In the CLA context, the minimum volatility portfolio is typically the final point reached by the algorithm, 
            representing the end of the critical line trace through the weight space.
          </Typography>
        </Paper>
        
        {/* Results Interpretation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting the Results
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Mean-Variance (MVO)
                </Typography>
                <Typography paragraph>
                  <strong>Portfolio Characteristics:</strong>
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Weights:</strong> Optimal balance between expected returns and volatility.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Asset Allocation:</strong> Typically more concentrated in higher-return assets compared to MinVol.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Risk Profile:</strong> Moderate risk with optimal risk-adjusted returns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Use Case:</strong> Investors seeking optimal risk-adjusted returns (highest Sharpe Ratio).
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Minimum Volatility (MinVol)
                </Typography>
                <Typography paragraph>
                  <strong>Portfolio Characteristics:</strong>
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Weights:</strong> Configured for lowest possible volatility portfolio.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Asset Allocation:</strong> More diversified across uncorrelated or negatively correlated assets.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Risk Profile:</strong> Lowest possible risk, possibly with reduced returns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Use Case:</strong> Conservative investors or volatile market conditions seeking stability.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Interpreting Efficient Frontier Visualizations
            </Typography>
            <Typography paragraph>
              The efficient frontier generated by CLA provides valuable insights:
            </Typography>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Corner Portfolios:</strong> Each critical point represents a structural change in the optimal portfolio composition.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Asset Entry/Exit Points:</strong> Points where assets enter or leave the optimal portfolio, providing valuable insights into when specific securities become relevant.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Weight Dynamics:</strong> Understanding how asset weights change as you move along the frontier helps with sensitivity analysis and scenario planning.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Risk Increments:</strong> The spacing between critical points shows how quickly risk increases relative to return, helping with risk management decisions.
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Advantages Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages of Critical Line Algorithm
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
                      <strong>Precision:</strong> Accurately identifies all portfolio transitions along the efficient frontier, providing exact solutions rather than approximations.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Completeness:</strong> Traces the entire efficient frontier in one systematic process, not just isolated points.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Stability:</strong> Provides stable, explicitly computed solutions rather than purely numerical approximations that might be sensitive to starting values.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Insight:</strong> Clearly shows how portfolios shift under changing constraints or targets, offering deep structural insights.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Theoretical Soundness:</strong> Directly implements the foundational theory of modern portfolio management.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Asset Dynamics:</strong> Reveals exactly when assets enter or leave the optimal portfolio as risk/return preferences change.
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
                      <strong>Computational Complexity:</strong> More computationally intensive than some numerical methods, especially for large asset universes.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Input Sensitivity:</strong> Like all portfolio optimization methods, results depend on the quality of return and covariance estimates.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Assumption Dependencies:</strong> Based on assumptions of quadratic utility and normal distribution of returns that may not always hold in real markets.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Implementation Complexity:</strong> More complex to implement correctly than simpler optimization methods, requiring careful handling of numerical precision issues.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Limited Constraint Types:</strong> Best suited for linear equality and inequality constraints; more complex constraint types may require extensions to the algorithm.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Advanced Topics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advanced Topics in CLA
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Extensions to Handle More Complex Constraints
          </Typography>
          <Typography paragraph>
            While the classic CLA handles linear equality and inequality constraints, several extensions exist to accommodate more complex scenarios:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Group Constraints:</strong> Limiting exposure to specific sectors or asset classes.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Transaction Cost Modeling:</strong> Incorporating trading costs into the optimization framework.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Turnover Constraints:</strong> Limiting portfolio changes to control rebalancing costs.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Factor Constraints:</strong> Managing exposure to specific risk factors beyond simple volatility.
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Relationship to Modern Extensions
          </Typography>
          <Typography paragraph>
            The CLA has inspired numerous modern portfolio optimization approaches:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Black-Litterman Model:</strong> Incorporates investor views alongside market equilibrium, using CLA for the optimization step.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Resampled Efficiency:</strong> Addresses estimation error by applying CLA to multiple simulated scenarios.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Robust Optimization:</strong> Extends CLA concepts to explicitly account for parameter uncertainty.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Multi-Period Optimization:</strong> Applies CLA concepts across time, considering intertemporal constraints and objectives.
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Computational Considerations
          </Typography>
          <Typography paragraph>
            Implementing CLA efficiently involves several technical considerations:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Numerical Stability:</strong> Careful handling of matrix operations to avoid accumulation of floating-point errors.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Degeneracy Handling:</strong> Techniques for dealing with degenerate cases where multiple constraints become active simultaneously.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Accelerated Computation:</strong> Methods for efficiently calculating critical points without full matrix reinversion at each step.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Parallelization:</strong> Approaches for parallelizing components of the algorithm to handle large asset universes.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Markowitz, H. M. (1952)</strong>. "Portfolio Selection." <em>Journal of Finance</em>, 7(1), 77-91.
                <MuiLink href="https://doi.org/10.1111/j.1540-6261.1952.tb01525.x" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Markowitz, H. M. (1956)</strong>. "The Optimization of a Quadratic Function Subject to Linear Constraints." <em>Naval Research Logistics Quarterly</em>, 3(1-2), 111-133.
                <MuiLink href="https://doi.org/10.1002/nav.3800030110" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Markowitz, H. M. (1987)</strong>. "Mean-Variance Analysis in Portfolio Choice and Capital Markets." Blackwell, Oxford, UK.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Bailey, D. H. & Lopez de Prado, M. (2013)</strong>. "An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization." <em>Algorithms</em>, 6(1), 169-196.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Niedermayer, A. & Niedermayer, D. (2010)</strong>. "Applying Markowitz's Critical Line Algorithm." <em>Handbook of Portfolio Construction</em>, 383-400.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Michaud, R. O. (1989)</strong>. "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?" <em>Financial Analysts Journal</em>, 45(1), 31-42.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Mean-Variance Optimization
                </Typography>
                <Typography variant="body2" paragraph>
                  The cornerstone of Modern Portfolio Theory that balances return and risk.
                </Typography>
                <Link href="/docs/mvo" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Minimum Volatility
                </Typography>
                <Typography variant="body2" paragraph>
                  Portfolio optimization approach focused solely on minimizing risk without a specific return target.
                </Typography>
                <Link href="/docs/min-vol" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Efficient Frontier
                </Typography>
                <Typography variant="body2" paragraph>
                  The set of optimal portfolios that offer the highest expected return for a defined level of risk.
                </Typography>
                <Link href="/docs/efficient-frontier" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

// Add getStaticProps to generate this page at build time
export const getStaticProps = async () => {
  return {
    props: {},
  };
};

export default CriticalLineAlgorithmPage; 