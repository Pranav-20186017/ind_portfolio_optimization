import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Button,
  Link as MuiLink
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

const MeanVarianceOptimizationPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Mean-Variance Optimization (MVO) | Portfolio Optimization</title>
        <meta name="description" content="Learn about Mean-Variance Optimization (MVO), a cornerstone of Modern Portfolio Theory that helps investors construct optimal portfolios." />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
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
            Mean-Variance Optimization (MVO)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            The cornerstone of Modern Portfolio Theory
          </Typography>
        </Box>
        
        {/* Introduction */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Introduction
          </Typography>
          <Typography paragraph>
            Mean-Variance Optimization (MVO), developed by Nobel laureate <strong>Harry Markowitz</strong>,
            forms the cornerstone of Modern Portfolio Theory (MPT). Introduced in his groundbreaking paper,
            "Portfolio Selection," published in the <em>Journal of Finance</em> in 1952, this method
            revolutionized how investors approach portfolio construction.
          </Typography>
          <Typography paragraph>
            Markowitz demonstrated that investors could optimize their portfolios by considering not just
            the expected returns of individual assets, but also how these assets move in relation to each other.
            This insight led to the formalization of diversification benefits through mathematical modeling,
            allowing investors to construct portfolios with superior risk-return characteristics.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Imagine going grocery shopping with a fixed budget. You want nutritious food (high returns) 
            but also want to avoid overspending or wasting money on overpriced items (minimizing risk). 
            MVO similarly helps investors select the best possible combination of assets (stocks, bonds, ETFs) 
            that give maximum possible returns while controlling risk exposure efficiently.
          </Typography>
          <Typography paragraph>
            The key insight of MVO is that combining assets that don't move in perfect sync (correlation less than 1)
            can actually reduce the overall risk of your portfolio while maintaining returns. This is the mathematical
            formalization of the old adage: "Don't put all your eggs in one basket."
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Example:</strong> Consider two stocks: a solar energy company and an umbrella manufacturer.
              When it's sunny, the solar company performs well but umbrella sales drop. When it's rainy,
              umbrella sales surge while solar energy production declines. By investing in both companies,
              your portfolio becomes more stable across weather conditions, even though each individual company
              experiences significant fluctuations.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical Explanation
          </Typography>
          <Typography paragraph>
            Mean-Variance Optimization mathematically balances expected returns against the volatility (risk) 
            of a portfolio. The optimization problem can be precisely formulated using the following notation:
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Notation:</strong>
            </Typography>
            <Typography component="div" paragraph>
              <InlineMath math="\mu \in \mathbb{R}^n" />: Expected returns vector for <InlineMath math="n" /> assets
            </Typography>
            <Typography component="div" paragraph>
              <InlineMath math="\Sigma \in \mathbb{R}^{n \times n}" />: Covariance matrix of asset returns
            </Typography>
            <Typography component="div" paragraph>
              <InlineMath math="w \in \mathbb{R}^n" />: Portfolio weights vector
            </Typography>
            <Typography component="div" paragraph>
              <InlineMath math="\mu^* \in \mathbb{R}" />: Target portfolio return
            </Typography>
            <Typography component="div" paragraph>
              <InlineMath math="\mathbf{1} \in \mathbb{R}^n" />: Vector of ones
            </Typography>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Portfolio Return
          </Typography>
          <Typography paragraph>
            The expected return (<InlineMath math="\mu_p" />) is calculated as a weighted sum of each asset's expected return:
          </Typography>
          <Equation math="\mu_p = w^T\mu = \sum_{i=1}^{n} w_i \mu_i" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Portfolio Variance
          </Typography>
          <Typography paragraph>
            Portfolio variance (<InlineMath math="\sigma_p^2" />), a measure of risk, is computed by considering not only individual asset 
            volatility but also how assets move together (covariance):
          </Typography>
          <Equation math="\sigma_p^2 = w^T\Sigma w = \sum_{i=1}^{n}\sum_{j=1}^{n} w_i w_j \sigma_{ij}" />
          
          <Typography paragraph>
            Where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <strong><InlineMath math="w" /></strong>: Vector of portfolio weights (<InlineMath math="\sum w_i = 1" />)
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong><InlineMath math="\mu" /></strong>: Vector of expected returns of assets
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong><InlineMath math="\Sigma" /></strong>: Covariance matrix representing asset interrelationships
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong><InlineMath math="\sigma_{ij}" /></strong>: Covariance between assets <InlineMath math="i" /> and <InlineMath math="j" />
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Optimization Problem
          </Typography>
          <Typography paragraph>
            The MVO problem mathematically stated is:
          </Typography>
          <Equation math="\begin{aligned} \min_{w} \quad & w^T \Sigma w \\ \text{subject to} \quad & w^T \mu \geq \mu^* \\ & w^T \mathbf{1} = 1 \\ & w_i \geq 0 \quad \forall i = 1,\ldots,n \end{aligned}" />
          
          <Typography paragraph>
            This optimization seeks the portfolio weights <InlineMath math="w" /> that minimize the portfolio variance (risk) 
            while ensuring the expected return meets or exceeds the target <InlineMath math="\mu^*" />, 
            the weights sum to 1 (full investment), and no short-selling is allowed (weights are non-negative).
          </Typography>
        </Paper>
        
        {/* Efficient Frontier */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Efficient Frontier
          </Typography>
          <Typography paragraph>
            The Efficient Frontier graphically represents the optimal portfolios achieving the highest possible 
            return for any given level of risk or the lowest risk for a given return. By varying the target return 
            <InlineMath math="\mu^*" />, we can trace out this curve representing the set of optimal portfolios.
          </Typography>
          
          <Box sx={{ textAlign: 'center', my: 3 }}>
            {/* Placeholder for an image - in production, replace with actual image */}
            <Paper 
              elevation={0}
              sx={{ 
                width: '100%', 
                height: 300, 
                bgcolor: '#f0f0f0', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}
            >
              <Typography variant="body2" color="text.secondary">
                [Efficient Frontier Graph Placeholder]
              </Typography>
            </Paper>
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              The Efficient Frontier curve showing optimal portfolios
            </Typography>
          </Box>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Interpretation
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Points on the curve:</strong> Optimal portfolios that offer the best possible return for a given level of risk.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Points below:</strong> Suboptimal portfolios (too much risk for too little return).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Points above:</strong> Unachievable with the available assets (not feasible).
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Tangency Portfolio and CML */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Tangency Portfolio and Capital Market Line (CML)
          </Typography>
          <Typography paragraph>
            When a <strong>risk-free asset</strong> (e.g., Government Treasury) is available, the Efficient Frontier 
            expands into a straight line called the <strong>Capital Market Line (CML)</strong>.
          </Typography>
          
          <Typography variant="h6" gutterBottom>
            Tangency Portfolio
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                The point where the CML touches (is tangent to) the Efficient Frontier.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Represents the portfolio with the highest Sharpe ratio (best return per unit of risk).
              </Typography>
            </li>
          </ul>
          
          <Equation math="\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}" />
          
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Capital Market Line (CML)
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                Defines optimal portfolios combining the risk-free asset and the tangency portfolio.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                All investors should hold combinations of the risk-free asset and the tangency portfolio according to their risk preferences.
              </Typography>
            </li>
          </ul>
          
          <Equation math="R_p = R_f + \frac{R_m - R_f}{\sigma_m} \sigma_p" />
          
          <Typography paragraph sx={{ mt: 2 }}>
            Where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="R_p" />: Expected portfolio return
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="R_f" />: Risk-free rate
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="R_m" />: Expected return of the tangency portfolio
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\sigma_m" />: Standard deviation of the tangency portfolio
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\sigma_p" />: Standard deviation of the portfolio
              </Typography>
            </li>
          </ul>
          
          <Box sx={{ textAlign: 'center', my: 3 }}>
            {/* Placeholder for an image - in production, replace with actual image */}
            <Paper 
              elevation={0}
              sx={{ 
                width: '100%', 
                height: 300, 
                bgcolor: '#f0f0f0', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}
            >
              <Typography variant="body2" color="text.secondary">
                [CML & Tangency Portfolio Graph Placeholder]
              </Typography>
            </Paper>
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              The Capital Market Line touching the Efficient Frontier at the Tangency Portfolio
            </Typography>
          </Box>
        </Paper>
        
        {/* Example Calculation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Example Calculation
          </Typography>
          <Typography paragraph>
            Consider a simple portfolio with two assets with the following characteristics:
          </Typography>
          
          <TableContainer component={Paper} sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Asset</strong></TableCell>
                  <TableCell><strong>Expected Return</strong></TableCell>
                  <TableCell><strong>Standard Deviation</strong></TableCell>
                  <TableCell><strong>Correlation</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>A</TableCell>
                  <TableCell>10%</TableCell>
                  <TableCell>8%</TableCell>
                  <TableCell rowSpan={2} align="center">0.3</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>B</TableCell>
                  <TableCell>15%</TableCell>
                  <TableCell>15%</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph>
            The covariance between A and B is calculated as:
          </Typography>
          <Equation math="\text{Cov}(A,B) = 0.3 \times 0.08 \times 0.15 = 0.0036" />
          
          <Typography paragraph>
            The covariance matrix <InlineMath math="\Sigma" /> is:
          </Typography>
          <Equation math="\Sigma = \begin{bmatrix} 0.0064 & 0.0036 \\ 0.0036 & 0.0225 \end{bmatrix}" />
          
          <Typography paragraph>
            Using the expected returns vector <InlineMath math="\mu = [0.10, 0.15]^T" />, we can solve the MVO problem for various target returns.
          </Typography>
          
          <Typography paragraph>
            For example, to find the minimum variance portfolio:
          </Typography>
          
          <Typography paragraph>
            <strong>Step 1:</strong> Set up the portfolio variance formula:
          </Typography>
          <Equation math="\sigma_p^2 = w_A^2 \times 0.0064 + w_B^2 \times 0.0225 + 2 \times w_A \times w_B \times 0.0036" />
          
          <Typography paragraph>
            <strong>Step 2:</strong> Since <InlineMath math="w_A + w_B = 1" />, we can substitute <InlineMath math="w_B = 1 - w_A" /> and minimize:
          </Typography>
          <Equation math="\sigma_p^2 = w_A^2 \times 0.0064 + (1-w_A)^2 \times 0.0225 + 2 \times w_A \times (1-w_A) \times 0.0036" />
          
          <Typography paragraph>
            <strong>Step 3:</strong> Take the derivative with respect to <InlineMath math="w_A" /> and set equal to zero to find the value of <InlineMath math="w_A" /> that minimizes portfolio variance.
          </Typography>
          
          <Typography paragraph>
            The minimum variance portfolio in this example would have approximately 70% in Asset A and 30% in Asset B, 
            resulting in a portfolio with expected return of:
          </Typography>
          <Equation math="\mu_p = 0.7 \times 0.10 + 0.3 \times 0.15 = 0.115 \text{ or } 11.5\%" />
          
          <Typography paragraph>
            And portfolio variance of:
          </Typography>
          <Equation math="\sigma_p^2 = 0.7^2 \times 0.0064 + 0.3^2 \times 0.0225 + 2 \times 0.7 \times 0.3 \times 0.0036 \approx 0.0054" />
          
          <Typography paragraph>
            Which gives a portfolio standard deviation of:
          </Typography>
          <Equation math="\sigma_p = \sqrt{0.0054} \approx 0.0735 \text{ or } 7.35\%" />
          
          <Typography paragraph>
            This is lower than the standard deviation of either asset individually (8% and 15%), demonstrating 
            the power of diversification.
          </Typography>
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
                      <strong>Pioneering Framework</strong>: Provides a mathematical foundation for portfolio construction that has stood the test of time, earning Markowitz a Nobel Prize.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Diversification Benefits</strong>: Quantifies the risk-reduction benefits of combining assets with imperfect correlations.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Risk-Return Trade-off</strong>: Explicitly models the relationship between risk and return, allowing investors to select portfolios that match their risk tolerance.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Intuitive Visualization</strong>: The Efficient Frontier provides a clear graphical representation of portfolio optimization choices.
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
                      <strong>Input Sensitivity</strong>: Highly sensitive to estimation errors in expected returns, variances, and covariances, which can lead to unreliable results.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Concentrated Portfolios</strong>: Often produces extremely concentrated portfolios, sometimes placing large weights on assets with estimation errors.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Normal Distribution Assumption</strong>: Assumes asset returns follow a normal distribution, which fails to capture fat tails and skewness in actual market returns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Single-Period Model</strong>: Does not account for time-varying risk and return parameters or multi-period investment horizons.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Transaction Costs</strong>: Basic implementation ignores transaction costs, taxes, and liquidity constraints that exist in real markets.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Practical Improvements */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Practical Improvements to Basic MVO
          </Typography>
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Challenge</strong></TableCell>
                  <TableCell><strong>Solution Approach</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Input sensitivity</TableCell>
                  <TableCell>Robust optimization; shrinkage estimators (James-Stein, Ledoit-Wolf); Bayesian approaches (Black-Litterman model)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Extreme weights</TableCell>
                  <TableCell>Weight constraints; regularization penalties; portfolio resampling techniques</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Non-normal returns</TableCell>
                  <TableCell>Higher-moment optimization; semi-variance optimization; historical simulation approaches</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Single-period limitation</TableCell>
                  <TableCell>Multi-period optimization; dynamic programming approaches; rolling-window optimization</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Market frictions</TableCell>
                  <TableCell>Transaction cost constraints; tax-aware optimization; turnover restrictions</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Markowitz, H. (1952)</strong>. "Portfolio Selection." <em>The Journal of Finance</em>, 7(1), 77-91.
                <MuiLink href="https://doi.org/10.1111/j.1540-6261.1952.tb01525.x" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Markowitz, H. (1959)</strong>. <em>Portfolio Selection: Efficient Diversification of Investments</em>. John Wiley & Sons.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Michaud, R. (1989)</strong>. "The Markowitz Optimization Enigma: Is Optimized Optimal?" <em>Financial Analysts Journal</em>, 45(1), 31-42.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Black, F. & Litterman, R. (1992)</strong>. "Global Portfolio Optimization." <em>Financial Analysts Journal</em>, 48(5), 28-43.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>DeMiguel, V., Garlappi, L., & Uppal, R. (2009)</strong>. "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" <em>The Review of Financial Studies</em>, 22(5), 1915-1953.
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
            {/* These would link to other educational pages once created */}
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Minimum Volatility
                </Typography>
                <Typography variant="body2" paragraph>
                  A portfolio optimization approach that focuses solely on minimizing risk without a specific return target.
                </Typography>
                <Button variant="outlined" color="primary" disabled>Coming Soon</Button>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Maximum Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  Finding the portfolio that maximizes risk-adjusted returns using the Sharpe ratio.
                </Typography>
                <Button variant="outlined" color="primary" disabled>Coming Soon</Button>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Modern Portfolio Theory
                </Typography>
                <Typography variant="body2" paragraph>
                  The broader theoretical framework that encompasses MVO and other portfolio optimization approaches.
                </Typography>
                <Link href="/docs/modern-portfolio-theory" passHref>
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

export default MeanVarianceOptimizationPage; 