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

const MinCVaRPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Min-CVaR for Indian Stock Portfolios | QuantPort India Docs</title>
        <meta name="description" content="Apply Minimum Conditional Value at Risk (CVaR) optimization to Indian equities. Protect your NSE/BSE portfolio against extreme market risks and downside events." />
        <meta property="og:title" content="Min-CVaR for Indian Stock Portfolios | QuantPort India Docs" />
        <meta property="og:description" content="Apply Minimum Conditional Value at Risk (CVaR) optimization to Indian equities. Protect your NSE/BSE portfolio against extreme market risks and downside events." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/min-cvar" />
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
            Minimum Conditional Value at Risk (CVaR) Optimization
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A tail risk-focused approach to portfolio construction
          </Typography>
        </Box>
        
        {/* Concept in One Paragraph */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Concept Overview
          </Typography>
          <Typography paragraph>
            <strong>Minimum Conditional Value at Risk (Min-CVaR) optimization</strong> is a portfolio construction method that explicitly focuses on minimizing the expected loss in the worst-case scenarios. Unlike traditional Mean-Variance Optimization that uses standard deviation as a risk measure, Min-CVaR specifically targets the "tail risk" of a portfolio—the severe losses that occur in extreme market conditions beyond the Value at Risk (VaR) threshold. This approach is particularly valuable for risk-averse investors concerned about downside protection, extreme market events, and asymmetric return distributions.
          </Typography>
        </Paper>
        
        {/* What is CVaR? */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            What is Conditional Value at Risk (CVaR)?
          </Typography>
          <Typography paragraph>
            <strong>Conditional Value at Risk (CVaR)</strong>, also known as <strong>Expected Shortfall (ES)</strong>, measures the expected loss in the worst-case scenarios that exceed the Value at Risk (VaR) threshold. In other words, if VaR tells you "I'm 95% confident losses won't exceed X," then CVaR tells you "But if they do exceed X, the average loss would be Y."
          </Typography>
          
          <Grid container spacing={3} sx={{ mt: 2 }}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  How it Relates to VaR
                </Typography>
                <Typography paragraph>
                  <Link href="/docs/value-at-risk" passHref>
                    <MuiLink>Value at Risk (VaR)</MuiLink>
                  </Link> is a threshold representing a loss amount that won't be exceeded with a certain confidence level (e.g., 95%).
                </Typography>
                <Typography paragraph>
                  CVaR goes one step further by quantifying the <em>average</em> or <em>expected loss</em> when that VaR threshold is breached.
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  This makes CVaR a more comprehensive measure of tail risk than VaR alone.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Mathematical Relationship
                </Typography>
                <Typography paragraph>
                  If we define VaR<sub>α</sub> as the VaR at confidence level α, then:
                </Typography>
                <Equation math="\text{CVaR}_\alpha(X) = \mathbb{E}[X | X \geq \text{VaR}_\alpha(X)]" />
                <Typography variant="body2">
                  Where X represents the loss distribution, and α is typically 0.95 or 0.99.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Mathematical Formulation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Formulation
          </Typography>
          
          <Typography paragraph>
            The Min-CVaR optimization problem can be stated as follows:
          </Typography>
          
          <Equation math="\begin{aligned}
\min_{\mathbf{w}} \quad & \text{CVaR}_\alpha(R_p) \\
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
                <InlineMath math="R_p" /> is the portfolio return
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="\alpha" /> is the confidence level (typically 0.95 or 0.99)
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Linear Programming Formulation
          </Typography>
          <Typography paragraph>
            Rockafellar and Uryasev (2000) showed that CVaR optimization can be reformulated as a linear programming problem. For a discrete set of scenarios with index values from 1 to T, the optimization problem becomes:
          </Typography>
          
          <Equation math="\begin{aligned}
\min_{\mathbf{w}, \gamma, \mathbf{z}} \quad & \gamma + \frac{1}{(1-\alpha)T}\sum_{j=1}^T z_j \\
\text{s.t.} \quad & z_j \geq -\mathbf{r}_j^\top \mathbf{w} - \gamma, \quad j = 1,2,...,T \\
& z_j \geq 0, \quad j = 1,2,...,T \\
& \mathbf{w}^\top \mathbf{1} = 1 \\
& \mathbf{w} \geq \mathbf{0} \quad \text{(optional)}
\end{aligned}" />
          
          <Typography paragraph>
            Where:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <InlineMath math="\gamma" /> is the VaR at confidence level α
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="z_j" /> are auxiliary variables capturing the excess losses
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="\mathbf{r}_j" /> is the vector of returns for scenario j
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="T" /> is the number of historical scenarios or simulations
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
            In our portfolio optimization system, Min-CVaR optimization is implemented using the following approach:
          </Typography>
          
          <ol>
            <li>
              <Typography paragraph>
                <strong>Historical simulation method</strong> is used to estimate scenario returns based on actual historical data.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Confidence level specification</strong>: Default confidence level is set to 95%, meaning we focus on the worst 5% of outcomes.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Linear programming solver</strong>: The problem is solved efficiently using modern optimization libraries.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Additional constraints</strong>: The basic formulation can be enhanced with sector constraints, individual asset limits, and target return constraints.
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
                  <TableCell>Historical returns are aligned and cleaned to remove missing values.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Optimization method</TableCell>
                  <TableCell>Linear programming using the Rockafellar and Uryasev reformulation.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Default confidence level</TableCell>
                  <TableCell>95% (focusing on the worst 5% of outcomes).</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Results handling</TableCell>
                  <TableCell>Both portfolio weights and risk metrics (VaR, CVaR) are returned and displayed.</TableCell>
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
                      <strong>Coherent Risk Measure</strong>: Unlike VaR, CVaR is a coherent risk measure, satisfying properties such as subadditivity (the risk of a combined portfolio is at most the sum of individual risks).
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Tail Risk Focus</strong>: Explicitly addresses severe market downturns and black swan events that traditional optimization methods may overlook.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Non-normal Returns</strong>: Well-suited for assets with skewed or fat-tailed return distributions, like many financial assets in practice.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Computational Tractability</strong>: Can be efficiently solved using linear programming techniques.
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
                      <strong>Data Sensitivity</strong>: Requires sufficient historical data or accurate simulations to model extreme events reliably.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Potential for Concentration</strong>: Without additional constraints, may lead to concentrated portfolios in pursuit of tail risk minimization.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Return Trade-off</strong>: Like other risk-minimization approaches, may sacrifice expected returns if not balanced with return targets.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Confidence Level Selection</strong>: Results can be sensitive to the choice of confidence level (e.g., 95% vs. 99%).
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
                  Risk-Averse Investors
                </Typography>
                <Typography variant="body2">
                  Particularly suitable for investors with strong aversion to large drawdowns and extreme market events, such as pension funds, endowments, and retirees.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Non-normal Asset Classes
                </Typography>
                <Typography variant="body2">
                  Excellent for portfolios containing assets with skewed or fat-tailed distributions, such as certain alternative investments, options, or emerging markets.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Regulatory Compliance
                </Typography>
                <Typography variant="body2">
                  Useful for financial institutions subject to regulatory frameworks that specifically address tail risk, such as Basel III requirements for banks.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Real-world Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Real-world Applications and Examples
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Typography variant="h6" gutterBottom>
              Example: Traditional vs. Min-CVaR Portfolio
            </Typography>
            <Typography paragraph>
              Consider two portfolios during the 2008 financial crisis:
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Traditional MVO Portfolio:
                </Typography>
                <ul>
                  <li>Expected Return: 8% annually</li>
                  <li>Volatility (σ): 15%</li>
                  <li>95% VaR: -6.5%</li>
                  <li>95% CVaR: -10.2%</li>
                  <li>Maximum Drawdown (2008): -45%</li>
                </ul>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Min-CVaR Portfolio:
                </Typography>
                <ul>
                  <li>Expected Return: 6.5% annually</li>
                  <li>Volatility (σ): 12%</li>
                  <li>95% VaR: -4.8%</li>
                  <li>95% CVaR: -7.1%</li>
                  <li>Maximum Drawdown (2008): -32%</li>
                </ul>
              </Grid>
            </Grid>
            <Typography paragraph sx={{ mt: 2 }}>
              The Min-CVaR portfolio sacrificed 1.5% in expected annual returns but significantly reduced tail risk metrics and actual drawdown during a market crisis.
            </Typography>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Industry Adoption
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Investment Banks</strong>: Use Min-CVaR for risk management in trading operations and structured product portfolios.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Pension Funds</strong>: Increasingly adopt tail risk-focused optimization to protect against extreme market downturns that could compromise obligations to beneficiaries.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Hedge Funds</strong>: Employ CVaR in alternative investment strategies, especially those involving derivatives or leveraged positions.
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
                <strong>Rockafellar, R. T., & Uryasev, S. (2000)</strong>. "Optimization of conditional value-at-risk." <em>Journal of Risk</em>, 2, 21-42.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Krokhmal, P., Palmquist, J., & Uryasev, S. (2002)</strong>. "Portfolio optimization with conditional value-at-risk objective and constraints." <em>Journal of Risk</em>, 4, 43-68.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Alexander, G. J., & Baptista, A. M. (2004)</strong>. "A comparison of VaR and CVaR constraints on portfolio selection with the mean-variance model." <em>Management Science</em>, 50(9), 1261-1273.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Favre, L., & Galeano, J. A. (2002)</strong>. "Mean-modified value-at-risk optimization with hedge funds." <em>Journal of Alternative Investments</em>, 5(2), 21-25.
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
                  Value at Risk (VaR)
                </Typography>
                <Typography variant="body2" paragraph>
                  A statistical technique used to measure the level of financial risk within a portfolio over a specific time frame.
                </Typography>
                <Link href="/docs/value-at-risk" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Conditional Value at Risk (CVaR)
                </Typography>
                <Typography variant="body2" paragraph>
                  A risk measure that quantifies the expected loss in the worst-case scenarios beyond the VaR threshold.
                </Typography>
                <Link href="/docs/conditional-value-at-risk" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Mean-Variance Optimization
                </Typography>
                <Typography variant="body2" paragraph>
                  The traditional approach to portfolio optimization that balances expected return against variance as a risk measure.
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
                  A portfolio optimization approach that focuses on minimizing overall volatility rather than specific tail risks.
                </Typography>
                <Link href="/docs/min-vol" passHref>
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

export default MinCVaRPage; 