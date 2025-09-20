import Head from 'next/head';
import { useRouter } from 'next/router';
import { canonical, ogImage, SITE_URL, SEOOpts } from '../lib/seo';

export default function SEO({ title, description, path, image }: SEOOpts) {
  const { asPath } = useRouter();
  const url = canonical(path || asPath || '/');
  const img = ogImage(image);

  const noindex = process.env.NODE_ENV !== 'production';

  return (
    <Head>
      <title>{title}</title>
      <meta name="description" content={description} />
      <link rel="canonical" href={url} />
      <meta name="robots" content={noindex ? 'noindex,nofollow' : 'index,follow'} />

      {/* Open Graph */}
      <meta property="og:type" content="website" />
      <meta property="og:title" content={title} />
      <meta property="og:description" content={description} />
      <meta property="og:url" content={url} />
      <meta property="og:image" content={img} />

      {/* Twitter */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={title} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={img} />

      {/* JSON-LD: SoftwareApplication (site-wide) */}
      <script
        type="application/ld+json"
        // @ts-ignore
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'SoftwareApplication',
            name: 'Indian Portfolio Optimization',
            applicationCategory: 'FinanceApplication',
            operatingSystem: 'Web',
            url,
            offers: { '@type': 'Offer', price: '0', priceCurrency: 'INR' }
          })
        }}
      />
    </Head>
  );
}
