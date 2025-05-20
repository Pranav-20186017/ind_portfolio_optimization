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

const EfficientFrontierPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Efficient Frontier | Portfolio Optimization</title>
        <meta name="description" content="Learn about the Efficient Frontier, a visual frontier of optimal portfolios that deliver the highest expected return for every attainable level of risk." />
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
            Efficient Frontier
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A visual frontier of optimal portfolios with maximum returns for each risk level
          </Typography>
        </Box>
        
        {/* Core Idea */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Core Idea
          </Typography>
          <Typography paragraph>
            A <strong>visual frontier of optimal portfolios</strong> that deliver the <strong>highest expected return for every attainable level of risk</strong> (or the lowest risk for every expected return). First formulated by Harry Markowitz in 1952, the frontier is the crown-jewel output of Modern Portfolio Theory (MPT).
          </Typography>
        </Paper>
        
        {/* Intuitive Picture */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Picture
          </Typography>
          <Typography paragraph>
            Imagine plotting every feasible portfolio in <strong>risk–return space</strong> (risk = σ, return = μ).
            The cloud's <strong>upper-left boundary</strong> hooks upward like a ski-slope—those boundary points form the <strong>efficient frontier</strong>:
          </Typography>
          
          <Box component="ul" sx={{ pl: 4 }}>
            <Box component="li">
              <Typography>
                Below the line → <strong>sub-optimal</strong> (same risk, lower return).
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                To the right → <strong>sub-optimal</strong> (same return, higher risk).
              </Typography>
            </Box>
          </Box>
          
          <Typography paragraph sx={{ mt: 2 }}>
            Investors should always choose somewhere <strong>on</strong> the frontier; everything else wastes opportunity.
          </Typography>
          
          {/* Placeholder for an image */}
          <Box sx={{ textAlign: 'center', my: 3 }}>
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
                [Efficient Frontier Visualization]
              </Typography>
            </Paper>
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              The efficient frontier (blue curve) showing the optimal portfolios with highest return for each risk level
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Definition */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Definition
          </Typography>
          <Typography paragraph>
            Given
          </Typography>
          <Box component="ul" sx={{ pl: 4 }}>
            <Box component="li">
              <Typography>
                weights <InlineMath math="\mathbf{w}\in\mathbb{R}^N,\; \mathbf{1}^\top\mathbf{w}=1" />
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                expected returns <InlineMath math="\boldsymbol{\mu}" />
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                covariance <InlineMath math="\boldsymbol{\Sigma}" />
              </Typography>
            </Box>
          </Box>
          
          <Typography paragraph sx={{ mt: 2 }}>
            an <strong>efficient portfolio</strong> solves either form:
          </Typography>
          
          <Equation math="\begin{aligned} \min_{\mathbf{w}}\;& \mathbf{w}^{\!\top}\!\boldsymbol{\Sigma}\,\mathbf{w} \\ \text{s.t. }& \boldsymbol{\mu}^{\!\top}\mathbf{w}= \mu^*, \; \mathbf{1}^{\!\top}\mathbf{w}=1 \end{aligned} \qquad \Longleftrightarrow \qquad \max_{\mathbf{w}}\; \frac{\boldsymbol{\mu}^{\!\top}\mathbf{w}-\mu_f} {\sqrt{\mathbf{w}^{\!\top}\!\boldsymbol{\Sigma}\,\mathbf{w}}}" />
          
          <Typography paragraph>
            Varying the target return <InlineMath math="\mu^*" /> (or risk-aversion constant) traces the entire frontier.
          </Typography>
        </Paper>
        
        {/* Shapes and Special Points */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Shapes and Special Points
          </Typography>
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Point</strong></TableCell>
                  <TableCell><strong>Definition</strong></TableCell>
                  <TableCell><strong>Role</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Minimum-Variance Portfolio (MVP)</strong></TableCell>
                  <TableCell>Left-most tip (lowest σ imaginable)</TableCell>
                  <TableCell>Baseline "safest" risky mix</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Tangency Portfolio</strong></TableCell>
                  <TableCell>Highest <strong>Sharpe Ratio</strong>; where the <strong>Capital Allocation Line (CAL)</strong> touches the frontier</TableCell>
                  <TableCell>Optimal mix when a risk-free asset is allowed</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Any frontier point</strong></TableCell>
                  <TableCell>Unique trade-off of μ and σ</TableCell>
                  <TableCell>Chosen per investor's risk tolerance</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Interpretation Tips for Users */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpretation Tips
          </Typography>
          <Box component="ul" sx={{ pl: 4 }}>
            <Box component="li">
              <Typography>
                Moving <strong>north-west</strong> along the curve = higher return at lower risk ⇒ always desirable.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                Frontier <strong>steepness</strong> signals diversification gain—steeper slope means adding risk is richly paid.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                A portfolio <strong>below</strong> the frontier should be re-optimised or hedged; it is leaving money on the table.
              </Typography>
            </Box>
          </Box>
        </Paper>
        
        {/* Limitations & Practical Tweaks */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Limitations & Practical Tweaks
          </Typography>
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Limitation</strong></TableCell>
                  <TableCell><strong>Mitigation</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Estimation error in μ, Σ</strong></TableCell>
                  <TableCell>Shrinkage (Ledoit-Wolf), Bayesian means, robust optimisation</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Ignores higher moments</strong> (skew, kurtosis)</TableCell>
                  <TableCell>"Post-Modern" extensions; downside risk optimisers</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>No constraints in theory</strong></TableCell>
                  <TableCell>Introduce weight bounds, sector caps</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Single-period assumption</strong></TableCell>
                  <TableCell>Multi-period or re-balancing simulation</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Relationship to CAPM & CAL */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Relationship to CAPM & CAL
          </Typography>
          <Typography paragraph>
            Adding a <strong>risk-free rate</strong> produces a straight <strong>Capital Allocation Line</strong> from <InlineMath math="R_f" /> tangent to the frontier.
            That tangency point is the market (or optimal) portfolio under <Link href="/education/capm" passHref><MuiLink>CAPM</MuiLink></Link> assumptions, and all investor choices become linear blends of <InlineMath math="R_f" /> and that portfolio.
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
                <strong>Markowitz, H. (1952)</strong>. "Portfolio Selection." <em>The Journal of Finance</em>, 7(1), 77-91.
                <MuiLink href="https://doi.org/10.1111/j.1540-6261.1952.tb01525.x" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
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
                  Modern Portfolio Theory
                </Typography>
                <Typography variant="body2" paragraph>
                  The theoretical framework that established the efficient frontier concept and revolutionized investment management.
                </Typography>
                <Link href="/education/modern-portfolio-theory" passHref>
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
                  The mathematical technique used to calculate portfolios along the efficient frontier.
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
                  Capital Asset Pricing Model
                </Typography>
                <Typography variant="body2" paragraph>
                  A model that builds on the efficient frontier to describe the relationship between systematic risk and expected return.
                </Typography>
                <Link href="/education/capm" passHref>
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

export default EfficientFrontierPage; 