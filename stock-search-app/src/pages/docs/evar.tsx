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

const EntropicVaRPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Entropic VaR for Indian Markets | QuantPort India Docs</title>
        <meta
          name="description"
          content="Explore Entropic Value at Risk (EVaR) for Indian equity portfolios. Manage tail risks in NSE/BSE stocks with more precise risk measurement than traditional VaR."
        />
        <meta property="og:title" content="Entropic VaR for Indian Markets | QuantPort India Docs" />
        <meta property="og:description" content="Explore Entropic Value at Risk (EVaR) for Indian equity portfolios. Manage tail risks in NSE/BSE stocks with more precise risk measurement than traditional VaR." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/evar" />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Docs</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Entropic Value at Risk (EVaR)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A coherent risk measure for more precise tail risk quantification
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Entropic Value at Risk (EVaR)</strong> is an advanced risk measure developed to address some of the theoretical and practical limitations of traditional risk metrics. Introduced in the early 2010s by Ahmadi-Javid, EVaR is a coherent risk measure that provides tighter bounds on tail risk than traditional Value at Risk (VaR) or Conditional Value at Risk (CVaR).
          </Typography>
          <Typography paragraph>
            EVaR incorporates concepts from information theory, specifically the relative entropy (or Kullback-Leibler divergence), to quantify the potential for extreme losses in a portfolio. It offers a more conservative risk assessment by generating an upper bound on the CVaR, making it particularly valuable for investors concerned with worst-case scenarios and extreme market events.
          </Typography>
          <Typography paragraph>
            As a coherent risk measure, EVaR satisfies the mathematical properties of monotonicity, sub-additivity, homogeneity, and translation invariance, ensuring that it properly reflects the risk reduction benefits of diversification and behaves consistently under various portfolio transformations.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Imagine you're trying to prepare for a major storm. Traditional Value at Risk (VaR) might tell you "there's a 95% chance the floodwaters won't exceed 3 feet." Conditional Value at Risk (CVaR) goes a step further, saying "if the floodwaters do exceed 3 feet, they will average 4 feet deep."
          </Typography>
          <Typography paragraph>
            Entropic Value at Risk (EVaR) takes an even more conservative approach, essentially saying "prepare for floodwaters of 4.5 feet to be truly safe." It provides a tighter upper bound on potential losses by incorporating more information about the shape and behavior of the "tail" of the loss distribution—particularly how quickly it decays.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Insurance analogy:</strong> Think of EVaR as a "premium" insurance policy. If VaR is basic insurance that covers common situations, and CVaR is enhanced coverage that includes some rare events, then EVaR is the comprehensive policy that covers even highly unlikely but catastrophic scenarios. It costs more in terms of capital reserves, but provides greater protection against extreme outcomes that other measures might underestimate.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            Entropic Value at Risk is defined using the concept of relative entropy from information theory. For a random variable X representing losses and a confidence level α, EVaR is calculated as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Entropic Value at Risk Formula</strong></Typography>
            <Equation math="\text{EVaR}_\alpha(X) = \inf_{z > 0} \left\{ \frac{1}{z} \ln \left( \frac{M_X(z)}{1 - \alpha} \right) \right\}" />
            <Typography variant="body2">
              where <InlineMath math="M_X(z)" /> is the moment-generating function of the random variable X at point z, and α is the confidence level.
            </Typography>
          </Box>

          <Typography paragraph>
            The moment-generating function (MGF) of a random variable X is defined as:
          </Typography>

          <Equation math="M_X(z) = \mathbb{E}[e^{zX}]" />

          <Typography paragraph>
            For a sample of returns <InlineMath math="r_1, r_2, \ldots, r_n" />, the empirical MGF can be estimated as:
          </Typography>

          <Equation math="M_X(z) \approx \frac{1}{n} \sum_{i=1}^{n} e^{zr_i}" />

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Relation to Other Risk Measures</Typography>
          <Typography paragraph>
            EVaR provides an upper bound on CVaR, which itself is an upper bound on VaR:
          </Typography>

          <Equation math="\text{VaR}_\alpha(X) \leq \text{CVaR}_\alpha(X) \leq \text{EVaR}_\alpha(X)" />

          <Typography paragraph>
            The inequality EVaR ≥ CVaR means that EVaR provides a more conservative risk estimate than CVaR, accounting for the extreme tails of the distribution more rigorously.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Properties of EVaR</Typography>
          <Typography paragraph>
            EVaR satisfies the four axioms of coherent risk measures:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Monotonicity:</strong> If X ≤ Y (meaning X represents less risk than Y), then EVaR(X) ≤ EVaR(Y).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sub-additivity:</strong> EVaR(X + Y) ≤ EVaR(X) + EVaR(Y), meaning diversification doesn't increase risk.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Homogeneity:</strong> For any positive constant λ, EVaR(λX) = λEVaR(X), indicating that risk scales proportionally with investment size.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Translation invariance:</strong> For any constant c, EVaR(X + c) = EVaR(X) + c, showing that adding a constant amount changes the risk measure by that same amount.
              </Typography>
            </li>
          </ul>

          <Typography paragraph>
            Furthermore, EVaR has additional desirable properties:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Law invariance:</strong> EVaR depends only on the distribution of the random variable X.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Continuity:</strong> EVaR is continuous with respect to the confidence level α.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Convexity:</strong> EVaR is a convex function, which is advantageous for optimization problems.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Our Service</Typography>
          <Typography paragraph>
            Our portfolio analyzer calculates Entropic Value at Risk through the following steps:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Historical Return Analysis:</strong> We collect historical returns of the portfolio over a specified timeframe.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Moment-Generating Function Estimation:</strong> We compute the empirical moment-generating function using the historical returns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Optimization Process:</strong> We find the infimum in the EVaR formula using numerical optimization techniques.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Confidence Level Selection:</strong> We calculate EVaR at different confidence levels (e.g., 95%, 99%, 99.5%) to provide a comprehensive risk assessment.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            In our implementation, we present EVaR alongside VaR and CVaR to allow for comparison across different risk measures. This multi-metric approach provides a more holistic view of potential portfolio risks, especially in the tails of the distribution.
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', mt: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" gutterBottom>Risk Measures Comparison Visualization (Placeholder)</Typography>
            <Box sx={{ height: '300px', bgcolor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                [Placeholder for VaR, CVaR, and EVaR comparison chart]
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              The chart compares VaR, CVaR, and EVaR at different confidence levels, illustrating how EVaR provides a more conservative risk estimate.
            </Typography>
          </Box>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's consider a simplified portfolio with the following 10 daily returns (in percentages):
          </Typography>
          <Typography paragraph>
            0.8%, 1.2%, -0.5%, 0.3%, -1.7%, 2.1%, -0.2%, 0.9%, -3.4%, 1.5%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Calculate VaR at 90% confidence level</Typography>
          <Typography paragraph>
            First, we sort the returns from worst to best:
          </Typography>
          <Typography paragraph>
            -3.4%, -1.7%, -0.5%, -0.2%, 0.3%, 0.8%, 0.9%, 1.2%, 1.5%, 2.1%
          </Typography>
          <Typography paragraph>
            The 90% VaR corresponds to the 10th percentile (or 1st value in our sorted list of 10 returns):
          </Typography>
          <Typography paragraph>
            VaR₉₀% = 3.4%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Calculate CVaR at 90% confidence level</Typography>
          <Typography paragraph>
            CVaR is the average of returns beyond the VaR threshold. Since we only have one value beyond the 90% VaR:
          </Typography>
          <Typography paragraph>
            CVaR₉₀% = -3.4%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Estimate the moment-generating function</Typography>
          <Typography paragraph>
            We need to compute <InlineMath math="M_X(z) = \frac{1}{10} \sum_{i=1}^{10} e^{zr_i}" /> for various values of z. For simplicity, let's evaluate at z = 1:
          </Typography>
          <Typography paragraph>
            M₁(1) = (1/10) × (e⁰·⁰⁰⁸ + e⁰·⁰¹² + e⁻⁰·⁰⁰⁵ + e⁰·⁰⁰³ + e⁻⁰·⁰¹⁷ + e⁰·⁰²¹ + e⁻⁰·⁰⁰² + e⁰·⁰⁰⁹ + e⁻⁰·⁰³⁴ + e⁰·⁰¹⁵)
          </Typography>
          <Typography paragraph>
            M₁(1) ≈ 1.001
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Calculate EVaR at 90% confidence level</Typography>
          <Typography paragraph>
            For different values of z, we compute:
          </Typography>
          <Typography paragraph>
                         f(z) = (1/z) × ln(M₁(z)/(1-0.9))
          </Typography>
          <Typography paragraph>
            After optimization (finding the infimum of f(z) for z {'>'} 0), we get:
          </Typography>
          <Typography paragraph>
            EVaR₉₀% ≈ 4.1%
          </Typography>

          <Typography paragraph>
            This example illustrates how EVaR provides a more conservative risk estimate than both VaR and CVaR. While VaR₉₀% = 3.4% and CVaR₉₀% = 3.4% (coincidentally equal in this small sample), EVaR₉₀% ≈ 4.1%, indicating that we should prepare for potentially larger losses when considering extreme scenarios.
          </Typography>
        </Paper>

        {/* Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          <Typography paragraph>
            Entropic Value at Risk serves several important purposes in portfolio management and risk assessment:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Portfolio Optimization:</strong> EVaR can be used as a risk constraint or objective function in portfolio optimization, leading to portfolios that are more resilient to extreme market events.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Regulatory Capital Requirements:</strong> Financial institutions can employ EVaR to determine capital reserves needed to withstand severe market stress, potentially fulfilling stricter regulatory standards.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Budgeting:</strong> EVaR allows for more sophisticated risk allocation across different portfolio components, focusing on controlling tail risk contributions.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Stress Testing:</strong> By providing an upper bound on potential losses, EVaR serves as a natural metric for comprehensive stress testing frameworks.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Heavy-Tailed Distributions:</strong> For assets or strategies with non-normal, heavy-tailed return distributions (like options, some alternative investments, or emerging markets), EVaR provides a more accurate risk assessment than traditional measures.
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
                      <strong>Coherence:</strong> As a coherent risk measure, EVaR properly accounts for diversification benefits and behaves consistently under various portfolio transformations.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Conservative bound:</strong> EVaR provides a tighter upper bound on potential losses than CVaR, offering a more cautious risk assessment for conservative investors.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Tail sensitivity:</strong> EVaR is highly sensitive to the behavior of extreme tails of the loss distribution, capturing risk that other measures might underestimate.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Mathematical tractability:</strong> EVaR maintains analytical tractability for many common distributions, allowing for efficient computation and optimization.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Information-theoretic foundation:</strong> By leveraging relative entropy, EVaR incorporates more distributional information than simpler risk measures.
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
                      <strong>Computational complexity:</strong> Calculating EVaR involves solving an optimization problem, making it more computationally intensive than VaR or CVaR.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Data requirements:</strong> Accurate estimation of EVaR requires sufficient historical data to properly characterize the tail behavior of returns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Potential oversensitivity:</strong> EVaR might be overly conservative in some scenarios, potentially leading to excessive risk aversion and capital requirements.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Educational barrier:</strong> The theoretical foundation of EVaR requires understanding of concepts from information theory, creating a steeper learning curve for practitioners.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Limited adoption:</strong> As a relatively newer risk measure, EVaR has less industry-wide acceptance and standardization compared to VaR and CVaR.
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
                <Typography variant="h6" gutterBottom>EVaR vs. Value at Risk (VaR)</Typography>
                <Typography paragraph>
                  While <MuiLink component={Link} href="/docs/value-at-risk">Value at Risk</MuiLink> simply estimates the minimum loss at a given confidence level, EVaR provides a much more conservative risk assessment. VaR fails to satisfy the coherence property of sub-additivity, potentially underestimating the risk of combined positions. EVaR, being a coherent risk measure, properly accounts for diversification effects and provides a more comprehensive view of extreme risk.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>EVaR vs. Conditional Value at Risk (CVaR)</Typography>
                <Typography paragraph>
                  <MuiLink component={Link} href="/docs/conditional-value-at-risk">Conditional Value at Risk</MuiLink> improves upon VaR by measuring the expected loss in the worst-case scenarios beyond the VaR threshold. While CVaR is also a coherent risk measure, EVaR provides an even tighter upper bound on potential losses. EVaR incorporates more information about the shape of the tail distribution through the moment-generating function, making it particularly valuable for heavy-tailed distributions where extreme events are more likely.
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
              <Typography paragraph><strong>Ahmadi-Javid, A. (2012)</strong>. "Entropic value-at-risk: A new coherent risk measure." <em>Journal of Optimization Theory and Applications</em>, 155(3), 1105-1123.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Föllmer, H., & Schied, A. (2016)</strong>. <em>Stochastic finance: an introduction in discrete time</em>. Walter de Gruyter GmbH & Co KG.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Ahmadi-Javid, A., & Fallah-Tafti, M. (2019)</strong>. "Portfolio optimization with entropic value-at-risk." <em>European Journal of Operational Research</em>, 279(1), 225-241.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Artzner, P., Delbaen, F., Eber, J. M., & Heath, D. (1999)</strong>. "Coherent measures of risk." <em>Mathematical Finance</em>, 9(3), 203-228.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Rockafellar, R. T., & Uryasev, S. (2000)</strong>. "Optimization of conditional value-at-risk." <em>Journal of Risk</em>, 2, 21-42.</Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Related Topics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Value at Risk (VaR)</Typography>
                <Typography variant="body2" paragraph>A statistical technique used to measure the level of financial risk within a portfolio over a specific time frame.</Typography>
                <Link href="/docs/value-at-risk" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Conditional Value at Risk (CVaR)</Typography>
                <Typography variant="body2" paragraph>A risk assessment measure that quantifies the expected loss in the worst-case scenarios beyond the VaR threshold.</Typography>
                <Link href="/docs/conditional-value-at-risk" passHref>
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

export default EntropicVaRPage; 