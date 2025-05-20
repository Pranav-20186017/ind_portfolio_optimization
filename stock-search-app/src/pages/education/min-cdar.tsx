import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
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

const MinCDaRPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Minimum Conditional Drawdown at Risk (CDaR) | Portfolio Optimization</title>
        <meta name="description" content="Learn about Minimum Conditional Drawdown at Risk (CDaR) portfolio optimization, an approach that minimizes the maximum expected drawdown with a certain confidence level." />
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
            Minimum Conditional Drawdown at Risk (CDaR)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Portfolio optimization focusing on minimizing severe drawdowns
          </Typography>
        </Box>
        
        {/* Concept Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Concept Overview
          </Typography>
          <Typography paragraph>
            <strong>Minimum Conditional Drawdown at Risk (Min-CDaR)</strong> is a portfolio optimization approach designed specifically to minimize the expected severity of drawdowns at a chosen confidence level. Unlike traditional variance-based methods that penalize both upside and downside movements, Min-CDaR directly targets portfolio drawdowns—the peak-to-trough declines experienced during market downturns. This makes it particularly valuable for investors who are sensitive to temporary capital depreciation and recovery periods, rather than short-term volatility.
          </Typography>
        </Paper>
        
        {/* Understanding Drawdowns */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Understanding Drawdowns
          </Typography>
          <Typography paragraph>
            <strong>Drawdown</strong> measures the decline from a historical peak to a subsequent trough in the value of a portfolio. It captures both the magnitude and duration of downward movements, making it an intuitive risk metric for practitioners.
          </Typography>
          
          <Grid container spacing={3} sx={{ mt: 2 }}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Drawdown Definition
                </Typography>
                <Typography paragraph>
                  For a portfolio with value process V(t), the drawdown at time t is defined as:
                </Typography>
                <Equation math="D(t) = \max_{0 \leq \tau \leq t} \frac{V(\tau) - V(t)}{V(\tau)}" />
                <Typography variant="body2">
                  This represents the percentage decline from the highest previous peak to the current value.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Maximum Drawdown
                </Typography>
                <Typography paragraph>
                  Maximum Drawdown (MDD) is the largest drawdown observed over a specific time period:
                </Typography>
                <Equation math="\text{MDD} = \max_{0 \leq t \leq T} D(t)" />
                <Typography variant="body2">
                  While MDD focuses on the single worst event, CDaR considers the expected severity of all drawdowns exceeding a threshold.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* From Drawdown to CDaR */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            From Drawdown to CDaR
          </Typography>
          <Typography paragraph>
            <strong>Conditional Drawdown at Risk (CDaR)</strong> extends the drawdown concept by focusing on the expected value of drawdowns that exceed a threshold at a specified confidence level. This parallels the relationship between Value at Risk (VaR) and Conditional Value at Risk (CVaR), but in drawdown space.
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Typography variant="h6" gutterBottom>
              Drawdown at Risk (DaR) vs. Conditional Drawdown at Risk (CDaR)
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Drawdown at Risk (DaR):
                </Typography>
                <Typography paragraph>
                  A threshold value that drawdowns will not exceed with a certain probability α (e.g., 95%). Comparable to VaR but for drawdowns.
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Conditional Drawdown at Risk (CDaR):
                </Typography>
                <Typography paragraph>
                  The expected value of drawdowns that exceed the DaR threshold. This represents the average severity of the worst drawdowns in the distribution.
                </Typography>
              </Grid>
            </Grid>
            <Equation math="\text{CDaR}_\alpha = \mathbb{E}[D(t) | D(t) \geq \text{DaR}_\alpha]" />
          </Box>
        </Paper>
        
        {/* Mathematical Formulation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Formulation
          </Typography>
          
          <Typography paragraph>
            The Min-CDaR optimization problem can be stated as follows:
          </Typography>
          
          <Equation math="\begin{aligned}
\min_{\mathbf{w}} \quad & \text{CDaR}_\alpha(\mathbf{w}) \\
\text{s.t.} \quad & \mathbf{w}^\top \mathbf{1} = 1 \\
& \mathbf{w} \geq \mathbf{0} \quad \text{(optional non-negativity constraint)}
\end{aligned}" />
          
          <Typography paragraph>
            Where:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <InlineMath math="\mathbf{w}" /> is the vector of portfolio weights
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="\alpha" /> is the confidence level (typically 0.90, 0.95, or 0.99)
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Linear Programming Formulation
          </Typography>
          <Typography paragraph>
            Similar to CVaR optimization, CDaR can be reformulated as a linear programming problem. For a discrete set of time points with observed portfolio values, the optimization becomes:
          </Typography>
          
          <Equation math="\begin{aligned}
\min_{\mathbf{w}, \gamma, \mathbf{u}, \mathbf{z}} \quad & \gamma + \frac{1}{(1-\alpha)T}\sum_{t=1}^T z_t \\
\text{s.t.} \quad & u_t \geq V(\tau, \mathbf{w}) - V(t, \mathbf{w}), \quad \forall \tau \leq t \\
& z_t \geq u_t - \gamma, \quad t = 1,2,...,T \\
& z_t \geq 0, \quad t = 1,2,...,T \\
& \mathbf{w}^\top \mathbf{1} = 1 \\
& \mathbf{w} \geq \mathbf{0} \quad \text{(optional)}
\end{aligned}" />
          
          <Typography paragraph>
            Where:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <InlineMath math="V(t, \mathbf{w})" /> is the portfolio value at time t given weights w
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="u_t" /> are auxiliary variables representing the drawdown at time t
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="z_t" /> are auxiliary variables capturing excess drawdowns beyond the threshold
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="\gamma" /> is the DaR at confidence level α
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="T" /> is the number of time points in the historical sample
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Implementation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Our Implementation
          </Typography>
          
          <Typography paragraph>
            In our portfolio optimization system, Min-CDaR optimization is implemented using the following approach:
          </Typography>
          
          <ol>
            <li>
              <Typography paragraph>
                <strong>Historical data processing</strong>: Calculate a time series of portfolio values based on historical asset returns and candidate portfolio weights.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Drawdown calculation</strong>: For each time point, compute the drawdown from the previous peak.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Linear programming formulation</strong>: Convert the CDaR minimization into a linear program that can be efficiently solved.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Additional constraints</strong>: The basic formulation can be enhanced with sector constraints, individual asset limits, or target return thresholds.
              </Typography>
            </li>
          </ol>
          
          <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>
            <strong>Key Implementation Details:</strong>
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Aspect</strong></TableCell>
                  <TableCell><strong>Detail</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Data preparation</TableCell>
                  <TableCell>Historical returns are converted to cumulative portfolio values to calculate drawdowns.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Optimization method</TableCell>
                  <TableCell>Linear programming using auxiliary variables to handle the path-dependent drawdown calculation.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Default confidence level</TableCell>
                  <TableCell>95% (focusing on the worst 5% of drawdowns).</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Results reporting</TableCell>
                  <TableCell>Both portfolio weights and risk metrics (DaR, CDaR, expected maximum drawdown) are calculated and displayed.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
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
                      <strong>Directly Targets Investor Experience</strong>: Focuses on drawdowns, which directly affect investor psychology and often trigger emotional decisions.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Path-Dependent Risk Measure</strong>: Unlike variance, CDaR accounts for the sequence and persistence of losses over time.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Recovery Periods</strong>: Indirectly addresses the time to recover from losses, which is critical for investors with specific time horizons.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>No Distribution Assumptions</strong>: Works with empirical return distributions without assuming normality or other specific distributions.
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
                      <strong>Computational Complexity</strong>: More computationally intensive than traditional MVO due to path-dependency and the need to calculate drawdowns over time.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Data Requirements</strong>: Needs substantial historical data to capture meaningful drawdown patterns and rare events.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>May Sacrifice Returns</strong>: Like other risk-minimization approaches, can result in portfolios with lower expected returns if not constrained.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time Period Sensitivity</strong>: Results can vary significantly based on the historical time period used for optimization.
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
            Ideal Use Cases
          </Typography>
          
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Long-Term Investors
                </Typography>
                <Typography variant="body2">
                  Well-suited for pension funds, endowments, and individual retirement portfolios where large drawdowns can seriously impact long-term objectives.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Target-Date Investments
                </Typography>
                <Typography variant="body2">
                  Excellent for investments with specific time horizons where recovery time from drawdowns becomes increasingly important as the target date approaches.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Drawdown-Sensitive Products
                </Typography>
                <Typography variant="body2">
                  Useful for financial products with explicit guarantees against drawdowns or those marketed as "drawdown-controlled" investments.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Comparison with Other Methods */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Comparison with Other Optimization Methods
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Optimization Method</strong></TableCell>
                  <TableCell><strong>Risk Measure</strong></TableCell>
                  <TableCell><strong>Key Differences from Min-CDaR</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Mean-Variance (MVO)</TableCell>
                  <TableCell>Variance (σ²)</TableCell>
                  <TableCell>Penalizes both upside and downside movements; ignores sequential losses</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Minimum Volatility</TableCell>
                  <TableCell>Standard deviation (σ)</TableCell>
                  <TableCell>Similar to MVO but without explicit return targets; ignores path dependency</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Minimum CVaR</TableCell>
                  <TableCell>Conditional Value at Risk</TableCell>
                  <TableCell>Focuses on tail losses in the return distribution rather than drawdowns over time</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Maximum Diversification</TableCell>
                  <TableCell>Diversification Ratio</TableCell>
                  <TableCell>Focuses on correlation structure rather than explicit risk minimization</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Real-world Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Real-world Applications and Examples
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Typography variant="h6" gutterBottom>
              Example: Traditional vs. Min-CDaR Portfolio
            </Typography>
            <Typography paragraph>
              Consider two portfolios during the 2008-2009 financial crisis:
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Traditional MVO Portfolio:
                </Typography>
                <ul>
                  <li>Expected Return: 7.5% annually</li>
                  <li>Volatility (σ): 14%</li>
                  <li>Maximum Drawdown: -48%</li>
                  <li>Time to Recovery: 37 months</li>
                </ul>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Min-CDaR Portfolio:
                </Typography>
                <ul>
                  <li>Expected Return: 6.2% annually</li>
                  <li>Volatility (σ): 11%</li>
                  <li>Maximum Drawdown: -28%</li>
                  <li>Time to Recovery: 19 months</li>
                </ul>
              </Grid>
            </Grid>
            <Typography paragraph sx={{ mt: 2 }}>
              The Min-CDaR portfolio sacrificed 1.3% in expected annual return but significantly reduced the maximum drawdown and recovery time during a severe market downturn.
            </Typography>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Industry Applications
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Target-Date Funds</strong>: Use drawdown control to reduce sequence-of-returns risk as investors approach retirement age.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Absolute Return Funds</strong>: Employ Min-CDaR as part of their toolkit to provide more stable return patterns with controlled drawdowns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Parity Strategies</strong>: Some risk parity implementations incorporate drawdown management as a secondary objective.
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
                <strong>Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005)</strong>. "Drawdown measure in portfolio optimization." <em>International Journal of Theoretical and Applied Finance</em>, 8(01), 13-58.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Goldberg, L. R., & Mahmoud, O. (2017)</strong>. "Drawdown: From practice to theory and back again." <em>Mathematics and Financial Economics</em>, 11(3), 275-297.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Zabarankin, M., Pavlikov, K., & Uryasev, S. (2014)</strong>. "Capital asset pricing model (CAPM) with drawdown measure." <em>European Journal of Operational Research</em>, 234(2), 508-517.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Alexandrovich, C., Stanislav, U., & Michael, Z. (2003)</strong>. "Portfolio optimization with drawdown constraints." <em>Asset and Liability Management</em>, 263-278.
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
                  Minimum CVaR
                </Typography>
                <Typography variant="body2" paragraph>
                  A portfolio optimization method that minimizes expected losses in the worst-case scenarios beyond the VaR threshold.
                </Typography>
                <Link href="/education/min-cvar" passHref>
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
                  A portfolio optimization approach that focuses on minimizing overall volatility rather than specific drawdowns.
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
                  Value at Risk (VaR)
                </Typography>
                <Typography variant="body2" paragraph>
                  A statistical measure of the risk of loss for investments, representing the minimum loss at a specific confidence level.
                </Typography>
                <Link href="/education/value-at-risk" passHref>
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

export default MinCDaRPage; 