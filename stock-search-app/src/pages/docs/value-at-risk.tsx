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
  TableRow,
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

const ValueAtRiskPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Value-at-Risk for Indian Portfolios | QuantPort India Docs</title>
        <meta name="description" content="Quantify potential losses in your Indian stock portfolio with Value-at-Risk (VaR). Assess downside risk for NSE/BSE investments with statistical confidence." />
        <meta property="og:title" content="Value-at-Risk for Indian Portfolios | QuantPort India Docs" />
        <meta property="og:description" content="Quantify potential losses in your Indian stock portfolio with Value-at-Risk (VaR). Assess downside risk for NSE/BSE investments with statistical confidence." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/value-at-risk" />
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
            Value-at-Risk (VaR)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Quantifying potential losses with statistical confidence
          </Typography>
        </Box>
        
        {/* What VaR Answers */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            What VaR Answers
          </Typography>
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 3, fontStyle: 'italic', borderLeft: '4px solid #2196f3' }}>
            <Typography variant="h6">
              "With X% confidence, how much could I lose at most over one day?"
            </Typography>
          </Box>
          
          <Typography paragraph>
            VaR is the industry's headline risk metric: one number that turns a whole return distribution into a worst-case threshold.
          </Typography>
          
          <Box sx={{ pl: 3, mb: 2 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>VaR 95%</strong> → losses worse than this happen only <strong>5%</strong> of the time.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>VaR 90%</strong> → losses worse than this happen <strong>10%</strong> of the time.
                </Typography>
              </li>
            </ul>
          </Box>
          
          <Grid container spacing={3} sx={{ mt: 2 }}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%', bgcolor: '#f5f5f5' }}>
                <Typography variant="h6" gutterBottom align="center">
                  Visual Intuition
                </Typography>
                <Typography variant="body2">
                  If you imagine the return distribution as a histogram, VaR is simply cutting off the left tail at a specific point. Everything to the left of that cutoff represents the worst-case scenarios that the VaR is measuring.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%', bgcolor: '#fff3e0' }}>
                <Typography variant="h6" gutterBottom align="center">
                  Practical Interpretation
                </Typography>
                <Typography variant="body2">
                  If a $1 million portfolio has a one-day 95% VaR of $20,000, this means there's a 95% probability that the portfolio won't lose more than $20,000 in a single day—or equivalently, there's a 5% chance of losing more than $20,000.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Formal Definition */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Formal Definition
          </Typography>
          <Typography paragraph>
            For a return random-variable <InlineMath math="R" /> and confidence level <InlineMath math="c" />:
          </Typography>
          
          <Equation math="\operatorname{VaR}_{c} \;=\; -\;\inf\bigl\{\,x\;|\;F_R(x)\ge 1-c\,\bigr\}" />
          
          <Typography paragraph>
            With <strong>daily simple returns</strong> (your data), <InlineMath math="\operatorname{VaR}_{95}" /> is just the <strong>5th percentile</strong> (a negative number).
          </Typography>
          
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mt: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Mathematical Explanation:</strong>
            </Typography>
            <Typography variant="body2">
              The formula finds the threshold value <InlineMath math="x" /> where the probability of getting a return less than or equal to <InlineMath math="x" /> is exactly <InlineMath math="1-c" />. The negative sign in front converts this to a loss amount (making VaR typically positive in finance literature, though we keep it negative in our implementation).
            </Typography>
          </Box>
        </Paper>
        
        {/* How Your Backend Computes VaR */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            How Our Backend Computes VaR
          </Typography>
          <Typography paragraph>
            Inside <strong>srv.py → compute_custom_metrics()</strong>:
          </Typography>
          <Box 
            component="pre" 
            sx={{ 
              p: 2, 
              bgcolor: '#f5f5f5', 
              borderRadius: 1, 
              overflow: 'auto',
              fontFamily: 'monospace',
              fontSize: '0.875rem'
            }}
          >
{`var_95  = np.percentile(port_returns, 5)   # 5-th percentile
cvar_95 = port_returns[port_returns <= var_95].mean()

var_90  = np.percentile(port_returns, 10)  # 10-th percentile
cvar_90 = port_returns[port_returns <= var_90].mean()`}
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 4 }}>
            Key points
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Step</strong></TableCell>
                  <TableCell><strong>What happens</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Historical method</strong></TableCell>
                  <TableCell>Uses the empirical distribution – no distributional assumptions.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Daily horizon</strong></TableCell>
                  <TableCell>Returns are daily ⇒ VaR is "1-day". <em>(Annualise ≈ VaR × √252 if needed.)</em></TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Sign</strong></TableCell>
                  <TableCell>Output is <strong>negative</strong> (loss). In the results table you multiply by 100 to show "-2.35%".</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>CVaR (Expected Shortfall)</strong></TableCell>
                  <TableCell>Mean of the tail beyond VaR – gives the <em>average</em> loss in worst cases.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Interpretation at 95% vs 90% */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpretation at 95% vs 90%
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Metric</strong></TableCell>
                  <TableCell><strong>Meaning</strong></TableCell>
                  <TableCell><strong>Usage</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>VaR 95%</strong></TableCell>
                  <TableCell>Loss exceeded on <strong>1 trading day in 20</strong> (on average).</TableCell>
                  <TableCell>Standard regulatory benchmark.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>VaR 90%</strong></TableCell>
                  <TableCell>Loss exceeded on <strong>1 day in 10</strong>.</TableCell>
                  <TableCell>Less conservative – useful for daily P&L limits.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, fontStyle: 'italic', borderLeft: '4px solid #2196f3' }}>
            <Typography>
              In your results card both numbers appear, helping users see <strong>"moderate tails"</strong> (90%) vs <strong>"deep tails"</strong> (95%).
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Example
          </Typography>
          <Typography paragraph>
            Suppose 500 days of daily returns sorted ascending:
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Percentile</strong></TableCell>
                  <TableCell><strong>Return</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>5% (25th obs)</TableCell>
                  <TableCell><strong>-1.8%</strong></TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>10% (50th obs)</TableCell>
                  <TableCell><strong>-1.1%</strong></TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Box sx={{ pl: 3, mb: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <InlineMath math="\text{VaR}_{95} = -0.018" /> → <em>"95% of the time I lose less than 1.8%."</em>
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <InlineMath math="\text{CVaR}_{95}" /> might be -2.4% → <em>average loss <strong>when</strong> the 1.8% barrier is breached.</em>
                </Typography>
              </li>
            </ul>
          </Box>
          
          <Typography paragraph>
            This example illustrates how VaR gives you a threshold for losses, while CVaR (Conditional VaR) tells you what the average loss is when you exceed that threshold—providing a more complete picture of tail risk.
          </Typography>
        </Paper>
        
        {/* Alternative VaR Methods */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Alternative VaR Methods (for practitioners)
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Method</strong></TableCell>
                  <TableCell><strong>Formula / Idea</strong></TableCell>
                  <TableCell><strong>When to prefer</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Parametric (Variance–Covariance)</strong></TableCell>
                  <TableCell><InlineMath math="\text{VaR}_{c}= -(\mu + z_c \sigma)" /></TableCell>
                  <TableCell>Large samples, near-normal returns.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Monte-Carlo</strong></TableCell>
                  <TableCell>Simulate thousands of paths; pick percentile.</TableCell>
                  <TableCell>Non-linear pay-offs, derivatives.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Filtered Historical</strong></TableCell>
                  <TableCell>GARCH volatility-scaled resampling.</TableCell>
                  <TableCell>Volatility-clustering markets.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          <Typography paragraph>
            Our platform currently uses <strong>Historical VaR</strong> – transparent, easy to explain, and assumption-free.
          </Typography>
          
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Advantage of Historical Method:</strong>
            </Typography>
            <Typography variant="body2">
              The historical approach makes no assumptions about the shape of the return distribution, unlike the parametric method which typically assumes normality. This is especially important for financial returns which often exhibit fat tails (higher kurtosis) and skewness that normal distributions don't capture.
            </Typography>
          </Box>
        </Paper>
        
        {/* Best-Practice Tips */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Best-Practice Tips
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Tip</strong></TableCell>
                  <TableCell><strong>Why</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Report <strong>CVaR/ES</strong> alongside VaR</TableCell>
                  <TableCell>CVaR is coherent and captures tail magnitude (we already compute both).</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Use <strong>rolling windows</strong></TableCell>
                  <TableCell>Tail risk drifts; a 1-year rolling VaR plot spots regime changes.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Align VaR horizon with user need</TableCell>
                  <TableCell>Daily for trading desks, 10-day for regulators, monthly for asset allocators.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Beware of <strong>leverage & derivatives</strong></TableCell>
                  <TableCell>VaR on returns may understate notional draw-downs; scale appropriately.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Advantages and Limitations Section */}
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
                      <strong>Simplicity and intuitiveness:</strong> Condenses complex risk distributions into a single, easy-to-understand number representing maximum expected loss.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Universal applicability:</strong> Can be applied to virtually any portfolio of assets regardless of asset class or complexity.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Confidence level flexibility:</strong> Can be adjusted (90%, 95%, 99%) based on risk tolerance and specific application needs.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Regulatory acceptance:</strong> Widely adopted by financial institutions and required by regulators as a standard risk measurement tool.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Comparative framework:</strong> Provides a consistent basis for comparing risk across different portfolios, strategies, or time periods.
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
                      <strong>Tail blindness:</strong> Provides no information about the severity of losses beyond the VaR threshold, potentially masking catastrophic tail risks.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Non-coherent measure:</strong> Lacks mathematical subadditivity, meaning the VaR of a combined portfolio can be greater than the sum of individual VaRs.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Method sensitivity:</strong> Results can vary significantly depending on calculation approach (historical, parametric, Monte Carlo) and parameters.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Backward-looking bias:</strong> Historical VaR assumes the past distribution of returns accurately reflects future risks, which may not hold during regime changes.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Liquidity blindness:</strong> Standard VaR calculations don't account for market liquidity constraints that may amplify losses during stress periods.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Jorion, P. (2006)</strong>. "Value at Risk: The New Benchmark for Managing Financial Risk." 3rd Edition. McGraw-Hill.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Alexander, C. (2008)</strong>. "Market Risk Analysis, Volume IV: Value-at-Risk Models." Wiley.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Dowd, K. (2002)</strong>. "Measuring Market Risk." Wiley.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Danielsson, J. (2011)</strong>. "Financial Risk Forecasting." Wiley.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* How It Appears in Your App */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            How It Appears in Our App
          </Typography>
          <Box sx={{ pl: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Metric row</strong> in each optimisation card: <em>"VaR 95%: –2.35%, CVaR 95%: –2.98%"</em>
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Histogram</strong> overlay shows red dashed line at VaR (our distribution plot does this).
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Tooltip</strong>: <em>"Worst daily loss at 95% confidence over back-test period."</em> → links to this page.
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Conclusion */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography paragraph sx={{ fontStyle: 'italic' }}>
            Providing both VaR 95% and VaR 90% helps novice users grasp everyday vs. rare-event risk, while practitioners still see the exact empirical thresholds and CVaR tail averages our engine calculates.
          </Typography>
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
                  Skewness
                </Typography>
                <Typography variant="body2" paragraph>
                  Measure of distribution asymmetry that affects the shape of the left tail and thus VaR calculations.
                </Typography>
                <Link href="/docs/skewness" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Kurtosis
                </Typography>
                <Typography variant="body2" paragraph>
                  Measure of tail fatness that directly impacts VaR and is particularly important for non-normal distributions.
                </Typography>
                <Link href="/docs/kurtosis" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Sortino Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  Risk-adjusted measure that focuses on downside risk, complementary to VaR and CVaR analysis.
                </Typography>
                <Link href="/docs/sortino-ratio" passHref>
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

export default ValueAtRiskPage; 