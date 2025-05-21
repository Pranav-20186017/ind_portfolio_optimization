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

const ConditionalValueAtRiskPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Conditional Value-at-Risk (CVaR) | Portfolio Optimization</title>
        <meta name="description" content="Learn about Conditional Value-at-Risk (CVaR), also known as Expected Shortfall (ES), which measures the expected loss in the worst-case scenarios beyond the VaR threshold." />
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
            Conditional Value-at-Risk (CVaR)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Also known as Expected Shortfall (ES) — Measuring the magnitude of tail losses
          </Typography>
        </Box>
        
        {/* Why Go Beyond VaR */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Why Go Beyond VaR?
          </Typography>
          <Typography paragraph>
            <strong>VaR</strong> tells you <strong>where</strong> the loss tail begins.
            <strong>CVaR</strong> tells you <strong>how deep that tail goes.</strong>
          </Typography>
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 2, fontStyle: 'italic', borderLeft: '4px solid #2196f3' }}>
            <Typography>
              "If the 95% VaR barrier is broken, what is my <strong>average</strong> loss on those worst days?"
            </Typography>
          </Box>
          <Typography paragraph>
            Because it captures <em>magnitude</em> as well as <em>frequency</em>, CVaR is recognized by regulators (Basel/FRTB) and academics 
            as a <strong>coherent</strong> risk measure—unlike VaR, it satisfies sub-additivity.
          </Typography>
        </Paper>
        
        {/* Formal Definition */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Formal Definition
          </Typography>
          <Typography paragraph>
            For confidence level <InlineMath math={"c"} /> (e.g., 95%):
          </Typography>
          
          <Equation math={"\\text{CVaR}_{c} \\;=\\; \\mathbb{E}\\!\\left[\\, -R \\;\\middle|\\; R \\le -\\text{VaR}_{c}\\,\\right]"} />
          
          <Typography paragraph>
            This formula calculates the average of all returns that fall below the VaR threshold. In simpler terms, 
            it tells us the average loss we can expect when a VaR breach occurs. The result is typically expressed 
            as a positive number representing the magnitude of expected loss.
          </Typography>
        </Paper>
        
        {/* Backend Implementation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Backend Implementation (Historical Method)
          </Typography>
          
          <Box component="pre" sx={{ p: 2, bgcolor: '#f1f1f1', borderRadius: 1, overflow: 'auto', fontSize: '0.875rem' }}>
            <code>
              {"var_95  = np.percentile(port_returns, 5)\n" +
               "cvar_95 = port_returns[port_returns <= var_95].mean()\n\n" +
               "var_90  = np.percentile(port_returns, 10)\n" +
               "cvar_90 = port_returns[port_returns <= var_90].mean()"}
            </code>
          </Box>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Key characteristics:
          </Typography>
          
          <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Aspect</strong></TableCell>
                  <TableCell><strong>Detail</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Data</strong></TableCell>
                  <TableCell>Daily simple returns (<code>port_returns</code>)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Horizon</strong></TableCell>
                  <TableCell>1-day (matching VaR)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Technique</strong></TableCell>
                  <TableCell>Purely empirical (no distributional assumptions)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Sign</strong></TableCell>
                  <TableCell>Stored <strong>negative</strong> (loss) → displayed as e.g. <strong>–3.2%</strong></TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph>
            Both <code>cvar_95</code> and <code>cvar_90</code> are returned in every optimization card.
          </Typography>
        </Paper>
        
        {/* Interpretation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpretation
          </Typography>
          
          <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Metric</strong></TableCell>
                  <TableCell><strong>Meaning</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>CVaR 95%</strong></TableCell>
                  <TableCell><em>Average</em> loss on the worst <strong>5%</strong> of days</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>CVaR 90%</strong></TableCell>
                  <TableCell>Average loss on the worst <strong>10%</strong> of days</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography variant="h6" gutterBottom>
            Rule of thumb:
          </Typography>
          <Box component="ul" sx={{ pl: 4 }}>
            <Box component="li">
              <Typography>
                <strong>CVaR ≈ VaR</strong> → loss tail is thin.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>CVaR ≫ VaR</strong> → tail is <em>fat</em>; draw-downs escalate once VaR is breached.
              </Typography>
            </Box>
          </Box>
        </Paper>
        
        {/* CVaR vs VaR - REVISED SECTION */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Understanding CVaR vs VaR
          </Typography>
          
          <Typography paragraph>
            Think of VaR and CVaR as two complementary ways of understanding portfolio risk:
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 3, border: '1px solid #e0e0e0', borderRadius: 2, height: '100%', bgcolor: '#f9f9f9' }}>
                <Typography variant="h6" align="center" gutterBottom>
                  Value at Risk (VaR)
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <Box>
                    <Typography paragraph>
                      <strong>What it tells you:</strong> "I am X% confident my losses won't exceed this threshold on a given day."
                    </Typography>
                    <Typography paragraph>
                      <strong>Real-world analogy:</strong> Like a flood warning level on a river - it tells you when to start worrying, but not how bad the flooding might get.
                    </Typography>
                  </Box>
                  <Box sx={{ mt: 2 }}>
                    <Typography sx={{ bgcolor: '#f3f3f3', p: 1.5, borderRadius: 1 }}>
                      <strong>Mathematical property:</strong> Not coherent (adding two portfolios can increase risk)
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 3, border: '1px solid #e0e0e0', borderRadius: 2, height: '100%', bgcolor: '#f9f9f9' }}>
                <Typography variant="h6" align="center" gutterBottom>
                  Conditional VaR (CVaR)
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <Box>
                    <Typography paragraph>
                      <strong>What it tells you:</strong> "When worst-case scenarios occur, this is the average loss I can expect."
                    </Typography>
                    <Typography paragraph>
                      <strong>Real-world analogy:</strong> Like measuring the average depth of flooding after the river exceeds the warning level - it tells you how severe problems might be.
                    </Typography>
                  </Box>
                  <Box sx={{ mt: 2 }}>
                    <Typography sx={{ bgcolor: '#f3f3f3', p: 1.5, borderRadius: 1 }}>
                      <strong>Mathematical property:</strong> Coherent (diversification always reduces risk)
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3, p: 2, bgcolor: '#e3f2fd', borderRadius: 1 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Why you need both:</strong>
            </Typography>
            <Typography paragraph>
              VaR is intuitive and answers "How often might I face significant losses?", while CVaR answers the follow-up question "How bad could those losses be?" Together, they provide a more complete picture of portfolio risk.
            </Typography>
            <Typography>
              This is why modern risk management frameworks and regulations increasingly require both measures.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Variants */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Variants
          </Typography>
          
          <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Variant</strong></TableCell>
                  <TableCell><strong>Formula / Method</strong></TableCell>
                  <TableCell><strong>Use-case</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Parametric</strong></TableCell>
                  <TableCell><InlineMath math={"\\mu - \\sigma\\dfrac{\\phi(z_c)}{1-c}"} /> (Normal)</TableCell>
                  <TableCell>Quick estimate for near-Gaussian assets</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Cornish-Fisher</strong></TableCell>
                  <TableCell>Adjust z-score using skew/kurtosis</TableCell>
                  <TableCell>Mildly non-normal tails</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Monte Carlo</strong></TableCell>
                  <TableCell>Simulate paths; average worst <InlineMath math={"1-c"} /> fraction</TableCell>
                  <TableCell>Options & non-linear pay-offs</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Filtered ES</strong></TableCell>
                  <TableCell>GARCH-scaled historical returns</TableCell>
                  <TableCell>Volatility-clustering markets</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph>
            Your platform currently runs the <strong>Historical ES</strong> (transparent and distribution-free).
          </Typography>
        </Paper>
        
        {/* Best-Practice Tips */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Best-Practice Tips
          </Typography>
          
          <Box component="ul" sx={{ pl: 4 }}>
            <Box component="li">
              <Typography>
                <strong>Report alongside VaR.</strong> (Already done.)
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Use rolling CVaR.</strong> Tail thickness evolves; a 1-year rolling plot flags build-ups.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Stress back-tests</strong>: compare crisis-year CVaR vs. normal periods.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Scenario scaling</strong>: for multi-day horizon <InlineMath math={"h"} /> under i.i.d., scale by <InlineMath math={"\\sqrt{h}"} /> — but beware volatility clustering.
              </Typography>
            </Box>
          </Box>
        </Paper>
        
        {/* Presentation in Your SPA */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Presentation in the Application
          </Typography>
          
          <Box component="ul" sx={{ pl: 4 }}>
            <Box component="li">
              <Typography>
                <strong>Metric rows</strong>: "CVaR 95%: –3.2%, CVaR 90%: –2.5%".
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Histogram overlay</strong>: solid red line for CVaR complementing dashed VaR.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Tooltip</strong>: "Average loss <em>given</em> the worst X% days." → link here.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Risk colour coding</strong>: highlight CVaR values that exceed user-set draw-down tolerances.
              </Typography>
            </Box>
          </Box>
          
          <Typography paragraph>
            By surfacing CVaR 95% and 90% next to VaR, your platform equips users with a <strong>full picture of tail risk</strong>—frequency <em>and</em> severity—critical for robust risk management and capital planning.
          </Typography>
        </Paper>
        
        {/* Graphical Illustration */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Graphical Illustration
          </Typography>
          
          <Typography paragraph>
            The relationship between VaR and CVaR can be visualized on a return distribution:
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: '#f8f9fa', border: '1px dashed #ccc', borderRadius: 2, mb: 3, textAlign: 'center' }}>
            <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 2 }}>
              [Distribution of Returns Graph]
            </Typography>
            <Typography variant="body2">
              X-axis: Portfolio returns<br />
              Y-axis: Frequency<br /><br />
              Left tail shows VaR as the cutoff point, and CVaR as the average of all returns beyond that point
            </Typography>
          </Box>
          
          <Typography paragraph>
            This illustration helps investors understand that while VaR represents a threshold (a single point in the distribution), 
            CVaR represents the average of the entire extreme tail beyond that threshold, providing a more comprehensive view of 
            potential extreme losses.
          </Typography>
        </Paper>
        
        {/* Applications in Portfolio Optimization */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Applications in Portfolio Optimization
          </Typography>
          
          <Typography paragraph>
            CVaR isn't just a risk measurement tool—it can be directly used as an optimization objective:
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, bgcolor: '#e8f5e9', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Minimum-CVaR Portfolios
                </Typography>
                <Typography variant="body2">
                  Directly minimize the expected shortfall of a portfolio to create allocations specifically designed 
                  to protect against tail events. These portfolios typically have different compositions than minimum-variance 
                  portfolios, especially when returns are not normally distributed.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, bgcolor: '#e8f5e9', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  CVaR Constraints
                </Typography>
                <Typography variant="body2">
                  Add CVaR constraints to other optimization objectives (like maximum return or Sharpe ratio) to ensure 
                  portfolios don't exceed specific tail risk thresholds, creating more resilient allocations that can 
                  withstand extreme market conditions.
                </Typography>
              </Box>
            </Grid>
          </Grid>
          
          <Typography paragraph sx={{ mt: 3 }}>
            The mathematical advantage of CVaR in optimization is that it can be reformulated as a linear programming problem, 
            making it computationally tractable even for large portfolios with many constraints.
          </Typography>
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
                      <strong>Coherent risk measure:</strong> Unlike VaR, CVaR satisfies the mathematical property of subadditivity, meaning diversification always reduces risk.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Tail sensitivity:</strong> Provides insight into the severity of extreme losses, not just their frequency, allowing for better worst-case scenario planning.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Regulatory alignment:</strong> Increasingly preferred by financial regulators (Basel Committee) for its more comprehensive risk assessment capabilities.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Optimization friendly:</strong> Can be formulated as a linear programming problem, making it tractable for complex portfolio optimization problems.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Distribution agnostic:</strong> Historical CVaR makes no assumptions about return distributions, capturing non-normal behavior common in financial markets.
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
                      <strong>Complexity:</strong> Less intuitive for non-technical users compared to VaR, requiring additional explanation to communicate its meaning effectively.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Data requirements:</strong> Needs more historical data to estimate accurately, as it focuses on tail events which are, by definition, rare.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Estimation sensitivity:</strong> Highly sensitive to outliers in the historical dataset, as these directly affect the tail average calculation.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Computational intensity:</strong> Some advanced CVaR calculation methods (like Monte Carlo) require significantly more computational resources than simple VaR.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Over-conservatism:</strong> May lead to excessively conservative portfolio allocations when used as the sole optimization objective, potentially sacrificing returns.
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
          
          <Box component="ul" sx={{ pl: 4 }}>
            <Box component="li">
              <Typography>
                <strong>Basel Committee on Banking Supervision (2019)</strong> – <em>FRTB</em>: Expected Shortfall at 97.5% replaces VaR.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Artzner et al. (1999)</strong> – "Coherent Measures of Risk." <em>Mathematical Finance</em>, 9(3), 203-228.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Rockafellar & Uryasev (2002)</strong> – "Conditional Value-at-Risk for General Loss Distributions." <em>Journal of Banking & Finance</em>, 26, 1443-1471.
              </Typography>
            </Box>
          </Box>
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
                  A statistical technique that measures the level of financial risk within a portfolio over a specific time frame.
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
                  Minimum CVaR Optimization
                </Typography>
                <Typography variant="body2" paragraph>
                  A portfolio optimization method that minimizes the expected loss in the worst-case scenarios beyond the VaR threshold.
                </Typography>
                <Link href="/docs/min-cvar" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Conditional Drawdown at Risk (CDaR)
                </Typography>
                <Typography variant="body2" paragraph>
                  A related risk measure that focuses on the expected drawdown when exceeding a specific drawdown threshold.
                </Typography>
                <Link href="/docs/min-cdar" passHref>
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

export default ConditionalValueAtRiskPage; 