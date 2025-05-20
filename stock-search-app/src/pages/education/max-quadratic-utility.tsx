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

const MaxQuadraticUtilityPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Maximum Quadratic Utility Optimization | Portfolio Optimization</title>
        <meta name="description" content="Learn about Maximum Quadratic Utility optimization, a method that incorporates investor risk aversion directly into the portfolio optimization process." />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/education" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Education
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
            Maximum Quadratic Utility Optimization
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Balancing return and risk based on investor risk aversion
          </Typography>
        </Box>
        
        {/* Overview Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            Maximum Quadratic Utility optimization is a sophisticated portfolio allocation method derived from Modern Portfolio Theory (MPT). 
            Unlike Mean-Variance Optimization (MVO), which seeks a balance between risk and return explicitly set by the investor, 
            the Maximum Quadratic Utility method incorporates the investor's risk aversion directly into the optimization process.
          </Typography>
          <Typography paragraph>
            This method identifies the optimal portfolio that maximizes the investor's quadratic utility function—considering both expected 
            returns and portfolio volatility in a single integrated measure. The approach is particularly valuable for investors who want their 
            personal risk tolerance directly factored into the portfolio construction process.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Imagine you're deciding how fast to drive on a highway. You balance the benefit (arriving earlier) with the risk 
            (chance of an accident or speeding ticket). Your personal comfort with risk influences your speed decision—this is 
            analogous to your "risk aversion."
          </Typography>
          <Typography paragraph>
            In investing, Maximum Quadratic Utility optimization does precisely this: it automatically balances potential returns 
            (arriving faster) against investment risk (chance of losses), given your risk tolerance. It selects the portfolio with 
            the highest personal satisfaction (utility), specifically tuned to your risk preference.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Example:</strong> Two investors examine the same set of assets but have different risk tolerances. 
              Investor A is young with a long investment horizon and high risk tolerance (low risk aversion parameter). 
              Investor B is nearing retirement with low risk tolerance (high risk aversion parameter). The Maximum Quadratic 
              Utility approach would recommend different optimal portfolios to each investor, even though they're using the 
              same underlying assets and market data.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical Formulation
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Quadratic Utility Function
          </Typography>
          <Typography paragraph>
            Investors aim to maximize the following quadratic utility function:
          </Typography>
          <Equation math="U(w) = w^T \mu - \frac{\delta}{2} w^T \Sigma w" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="w" /> is the vector of portfolio weights.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\mu" /> is the vector of expected asset returns.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\Sigma" /> is the covariance matrix of asset returns.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\delta" /> (delta) is the investor's risk aversion parameter, reflecting how strongly the investor dislikes volatility.
              </Typography>
            </li>
          </ul>

          <Typography paragraph>
            The first term <InlineMath math="w^T \mu" /> represents the expected portfolio return. 
            The second term <InlineMath math="\frac{\delta}{2} w^T \Sigma w" /> represents a penalty for portfolio risk (volatility) scaled by the risk aversion parameter.
            The utility function combines expected returns and risk into a single value. Maximizing this utility gives the optimal trade-off 
            between return and risk specific to the investor's risk tolerance.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Mathematical Optimization Problem
          </Typography>
          <Typography paragraph>
            Formally, the optimization problem is:
          </Typography>
          <Equation math="\begin{aligned} & \underset{w}{\text{maximize}} & & w^T \mu - \frac{\delta}{2} w^T \Sigma w \\ & \text{subject to} & & \mathbf{1}^T w = 1 \\ & & & w_i \geq 0, \quad \forall i \end{aligned}" />
          
          <Typography variant="subtitle1" gutterBottom>
            <strong>Constraints</strong>
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <strong>Full investment constraint</strong>: <InlineMath math="\mathbf{1}^T w = 1" />, ensuring that all available funds are allocated (sum of weights equals 1).
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong>Non-negativity constraint</strong> (optional): <InlineMath math="w_i \geq 0, \forall i" />, preventing short selling.
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Solving the Optimization Problem
          </Typography>
          <Typography paragraph>
            This is a quadratic programming problem with linear constraints. When no other constraints are present besides the full investment constraint,
            there is an analytical solution:
          </Typography>
          <Equation math="w_{optimal} = \frac{1}{\delta} \Sigma^{-1} \mu" />
          <Typography paragraph>
            However, with additional constraints like the non-negativity constraint, the problem typically requires numerical optimization 
            methods such as quadratic programming solvers.
          </Typography>

          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Understanding the Risk Aversion Parameter
          </Typography>
          <Typography paragraph>
            The risk aversion parameter <InlineMath math="\delta" /> plays a crucial role in determining the optimal portfolio:
          </Typography>
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Low Risk Aversion (δ ≈ 1-3)</strong>: Higher tolerance to risk; portfolios favor higher returns. The optimization will place 
                  more emphasis on maximizing expected returns, potentially accepting higher volatility.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Moderate Risk Aversion (δ ≈ 4-6)</strong>: Balance between returns and volatility. This represents a more balanced approach,
                  seeking reasonable returns while maintaining moderate risk control.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>High Risk Aversion (δ {'>'} 6)</strong>: Lower-risk, more conservative portfolios. The optimization will strongly favor risk reduction,
                  even at the expense of potential returns.
                </Typography>
              </li>
            </ul>
          </Box>
          
          <Typography paragraph>
            As <InlineMath math="\delta" /> approaches infinity, the Maximum Quadratic Utility solution converges to the Minimum Variance portfolio, where risk minimization is the only concern.
            Conversely, as <InlineMath math="\delta" /> approaches zero, the solution would favor the highest possible return regardless of risk.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Relationship to the Efficient Frontier
          </Typography>
          <Typography paragraph>
            On the efficient frontier graph, portfolios that maximize quadratic utility at different risk aversion levels correspond to different points along the efficient frontier. 
            Each value of <InlineMath math="\delta" /> identifies a specific point on the efficient frontier that is optimal for an investor with that particular risk aversion.
          </Typography>
          <Equation math="U_{max}(\delta) \rightarrow \text{point on efficient frontier}" />
          <Typography paragraph>
            Graphically, this can be represented as a series of indifference curves (representing the same level of utility) tangent to the efficient frontier, 
            with the tangent point being the optimal portfolio for the given risk aversion.
          </Typography>
        </Paper>
        
        {/* Key Differences from Other Methods */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Comparison with Other Optimization Methods
          </Typography>
          
          <Box sx={{ mt: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Paper elevation={0} sx={{ bgcolor: '#f8f9fa', p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Mean-Variance Optimization
                  </Typography>
                  <ul>
                    <li><Typography paragraph>Requires specifying a target return or risk level</Typography></li>
                    <li><Typography paragraph>Produces points along the efficient frontier</Typography></li>
                    <li><Typography paragraph>Explicitly balances risk and return</Typography></li>
                    <li><Typography paragraph>Risk aversion is implicit in the choice of target</Typography></li>
                  </ul>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper elevation={0} sx={{ bgcolor: '#f1f8e9', p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Maximum Quadratic Utility
                  </Typography>
                  <ul>
                    <li><Typography paragraph>Explicitly incorporates risk aversion parameter</Typography></li>
                    <li><Typography paragraph>Combines risk and return in a single objective</Typography></li>
                    <li><Typography paragraph>Produces a single optimal portfolio for a given risk aversion</Typography></li>
                    <li><Typography paragraph>More intuitive for personal risk preferences</Typography></li>
                  </ul>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper elevation={0} sx={{ bgcolor: '#f3e5f5', p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Minimum Volatility
                  </Typography>
                  <ul>
                    <li><Typography paragraph>Focuses solely on minimizing risk</Typography></li>
                    <li><Typography paragraph>Equivalent to infinite risk aversion</Typography></li>
                    <li><Typography paragraph>Produces a single minimum risk portfolio</Typography></li>
                    <li><Typography paragraph>No explicit consideration of expected returns</Typography></li>
                  </ul>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        </Paper>
        
        {/* Properties Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Properties of Maximum Quadratic Utility Portfolios
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
                      <strong>Personalized Risk Management:</strong> Directly accounts for investor-specific risk preferences through the risk aversion parameter.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Flexible Optimization:</strong> Allows easy adjustments to reflect varying market conditions or changing investor risk tolerance.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Theoretically Sound:</strong> Grounded firmly in economic theory and utility maximization principles, providing intuitive and statistically meaningful results.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Efficient Decision Making:</strong> Combines both risk and return considerations into a single optimization decision.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Economic Rationale:</strong> Aligns with the concept that investors seek to maximize their personal utility rather than pursuing risk or return in isolation.
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
                      <strong>Risk Aversion Calibration:</strong> Determining the appropriate risk aversion parameter for an individual investor can be challenging.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Estimation Sensitivity:</strong> Like other optimization methods, remains sensitive to errors in estimating expected returns and covariances.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Quadratic Approximation:</strong> The quadratic utility function is only an approximation of true investor preferences, which may be more complex.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Assumes Normal Returns:</strong> Implicitly assumes returns follow a normal distribution, which may not capture extreme market events.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Complexity for Retail Investors:</strong> The abstract nature of risk aversion parameters may be difficult for some investors to conceptualize.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Use Cases */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Practical Applications and Use Cases
          </Typography>
          
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Personalized Portfolio Management
                </Typography>
                <Typography variant="body2">
                  Ideal for wealth managers and financial advisors who want to tailor portfolios to individual client risk preferences.
                  The risk aversion parameter can be adjusted based on client questionnaires or financial goals.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Robo-Advisory Platforms
                </Typography>
                <Typography variant="body2">
                  Automated investment services can map user risk profiles to specific risk aversion parameters, creating algorithmically 
                  tailored portfolios that match individual preferences without human intervention.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Dynamic Portfolio Rebalancing
                </Typography>
                <Typography variant="body2">
                  As market conditions or investor circumstances change, the risk aversion parameter can be adjusted to shift the 
                  portfolio dynamically between more aggressive and more conservative positions.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Risk Aversion Assessment */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Determining Your Risk Aversion Level
          </Typography>
          <Typography paragraph>
            Your risk aversion parameter is personal and depends on factors like age, financial situation, investment goals, and psychological comfort with risk.
            Consider these guidelines to help determine an appropriate value:
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom align="center">
                  Lower Risk Aversion (δ = 1-3)
                </Typography>
                <ul>
                  <li><Typography variant="body2">Longer investment horizon (10+ years)</Typography></li>
                  <li><Typography variant="body2">Stable income source</Typography></li>
                  <li><Typography variant="body2">Higher capacity to withstand losses</Typography></li>
                  <li><Typography variant="body2">Growth-focused investment objectives</Typography></li>
                  <li><Typography variant="body2">Comfortable with higher volatility</Typography></li>
                </ul>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom align="center">
                  Moderate Risk Aversion (δ = 4-6)
                </Typography>
                <ul>
                  <li><Typography variant="body2">Medium investment horizon (5-10 years)</Typography></li>
                  <li><Typography variant="body2">Moderately stable finances</Typography></li>
                  <li><Typography variant="body2">Balance between growth and preservation</Typography></li>
                  <li><Typography variant="body2">Some need for current income</Typography></li>
                  <li><Typography variant="body2">Moderate comfort with market fluctuations</Typography></li>
                </ul>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom align="center">
                  Higher Risk Aversion (δ = 7+)
                </Typography>
                <ul>
                  <li><Typography variant="body2">Shorter investment horizon (0-5 years)</Typography></li>
                  <li><Typography variant="body2">Approaching or in retirement</Typography></li>
                  <li><Typography variant="body2">Focus on capital preservation</Typography></li>
                  <li><Typography variant="body2">Strong need for current income</Typography></li>
                  <li><Typography variant="body2">Discomfort with portfolio volatility</Typography></li>
                </ul>
              </Grid>
            </Grid>
          </Box>
          
          <Typography paragraph>
            Note that your risk aversion may change over time as your circumstances evolve. Regular reassessment is recommended.
          </Typography>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Arrow, K. J. (1971)</strong>. "Essays in the Theory of Risk-Bearing." North-Holland Publishing Company.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sharpe, W. F. (1964)</strong>. "Capital Asset Prices: A Theory of Market Equilibrium Under Conditions of Risk." <em>Journal of Finance</em>, 19(3), 425-442.
                <MuiLink href="https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1964.tb02865.x" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Ledoit, O., & Wolf, M. (2004)</strong>. "A well-conditioned estimator for large-dimensional covariance matrices." 
                <em>Journal of Multivariate Analysis</em>, 88(2), 365-411.
                <MuiLink href="https://doi.org/10.1016/S0047-259X%2803%2900096-4" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Markowitz, H. M. (1952)</strong>. "Portfolio Selection." <em>The Journal of Finance</em>, 7, 77-91.
                <MuiLink href="https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1952.tb01525.x" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Pratt, J. W. (1964)</strong>. "Risk Aversion in the Small and in the Large." <em>Econometrica</em>, 32(1/2), 122-136.
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
                <Link href="/education/mvo" passHref>
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
                <Link href="/education/min-vol" passHref>
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
                <Button variant="outlined" color="primary" disabled>Coming Soon</Button>
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

export default MaxQuadraticUtilityPage; 