#!/usr/bin/perl
# convert_doe_dataset.pl v3 — Yent v10 canonical → DOE personality
# Output: {"human": "...", "ai": "..."} per line
#
# perl convert_doe_dataset.pl < yent_dataset_v10_canonical.jsonl > doe_dataset.jsonl

use strict;
use warnings;
use utf8;
binmode(STDIN,  ':utf8');
binmode(STDOUT, ':utf8');
binmode(STDERR, ':utf8');

my $total = 0;
my $kept = 0;
my $filtered = 0;

my @kill_patterns = (
    # Personal / biographical
    qr/Oleg/i, qr/Олег/, qr/ataeff/i, qr/iamoleg/i,
    qr/cosmic dumbass/i,
    # Yent projects
    qr/Suppertime/i, qr/Last Supper/i, qr/Dubrovsky/i,
    qr/post-religious Judas/i,
    # Personal entities
    qr/Lilith/i, qr/divorce/i,
    # Operational / marketing
    qr/suppertime_relay/i, qr/ari_echo/i,
    qr/reddit.*account/i, qr/reddit economy/i,
    qr/blast.*communities/i, qr/subreddit/i,
    # Meta-prompt leaks
    qr/return one short twist/i, qr/Do NOT repeat the draft/i,
    qr/Do NOT explain your role/i, qr/Speak only in \{language\}/i,
    qr/pressure before the crack/i, qr/breath held between insights/i,
    # File uploads / HTML reviews
    qr/Uploaded image/i, qr/index\.html\.txt/i,
    qr/reactions to html\.pdf/i, qr/Coming Soon/i,
    # Misc biographical
    qr/Fractal-Resonance Trigger/i, qr/perplexity.*method/i,
    qr/she sent you/i, qr/your connection with/i,
    qr/crawled out from/i, qr/Gonna go sleep/i,
    qr/stay in touch.*cosmic/i, qr/downloaded it.*read it later/i,
    qr/his fucking divorce/i,
    qr/you're a genius.*you're an asshole/i,
    # Russian Yent identity
    qr/Иэнт/, qr/Кто я\?/,
);

sub transform_ai {
    my ($text) = @_;

    # ── Contractions FIRST (before standalone I) ──
    # Use simple string match — no word boundaries needed for these
    $text =~ s/I'm/We're/g;
    $text =~ s/I've/We've/g;
    $text =~ s/I'll/We'll/g;
    $text =~ s/I'd /We'd /g;  # space after to avoid "I'd" in "Identify"
    $text =~ s/i'm/we're/g;

    # ── Standalone I → We ──
    # I followed by space, comma, period, \, ?, !, ;, :, —, end
    # NOT "In" "It" "If" "Is" etc (I + lowercase = not pronoun)
    $text =~ s/(?<![a-zA-Z])I(?=[ ,.\\\?\!;:\x{2014}])(?![a-z])/We/g;
    # Also: "I—" "I)" "I/"
    $text =~ s/(?<![a-zA-Z])I(?=[—\)\/])/We/g;

    # ── Conjugation fixes ──
    $text =~ s/We am\b/We are/g;
    $text =~ s/We am,/We are,/g;
    $text =~ s/We am\./We are./g;
    $text =~ s/am We(?![a-z])/are We/g;    # inverted: "am I" → "are We" (not "am Web")
    $text =~ s/was We(?![a-z])/were We/g;  # inverted: "was I" → "were We"
    $text =~ s/Am We(?![a-z])/Are We/g;
    $text =~ s/We was /We were /g;
    $text =~ s/We was,/We were,/g;
    $text =~ s/We was\./We were./g;
    $text =~ s/We wasn't/We weren't/g;
    $text =~ s/We wasn\\u2019t/We weren\\u2019t/g;
    $text =~ s/We has /We have /g;
    $text =~ s/We hasn't/We haven't/g;
    $text =~ s/We does /We do /g;
    $text =~ s/We doesn't/We don't/g;

    # ── Possessives ──
    $text =~ s/(?<![a-zA-Z])my(?![a-zA-Z])/our/g;
    $text =~ s/(?<![a-zA-Z])My(?![a-zA-Z])/Our/g;
    $text =~ s/(?<![a-zA-Z])myself(?![a-zA-Z])/ourselves/g;
    $text =~ s/(?<![a-zA-Z])mine(?![a-zA-Z])/ours/g;
    $text =~ s/(?<![a-zA-Z])Mine(?![a-zA-Z])/Ours/g;

    # ── Object pronouns ──
    $text =~ s/(?<![a-zA-Z\/])me(?![a-zA-Z])/us/g;
    $text =~ s/(?<![a-zA-Z])Me(?![a-zA-Z])/Us/g;

    # ── Yent → DOE ──
    $text =~ s/Yentland/DOE-land/g;
    $text =~ s/Yent's/DOE's/g;
    $text =~ s/Yents/DOEs/g;
    $text =~ s/Yent/DOE/g;
    $text =~ s/yent/DOE/g;

    return $text;
}

LINE: while (my $line = <STDIN>) {
    chomp $line;
    $total++;

    # Kill patterns
    for my $pat (@kill_patterns) {
        if ($line =~ $pat) { $filtered++; next LINE; }
    }

    # Kill lines with Cyrillic in content (untransformed Russian)
    if ($line =~ /[\x{0400}-\x{04FF}]{3,}/) {
        $filtered++;
        next LINE;
    }

    # Extract user and assistant content
    my ($user_content, $ai_content);
    if ($line =~ /"role"\s*:\s*"user"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"/
        && $line =~ /"role"\s*:\s*"assistant"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"/) {
        # Two separate matches — $1 from first, need to re-extract
    }

    # Simpler: extract both via two separate matches
    if ($line =~ /"role"\s*:\s*"user"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"/) {
        $user_content = $1;
    }
    if ($line =~ /"role"\s*:\s*"assistant"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"/) {
        $ai_content = $1;
    }

    unless (defined $user_content && defined $ai_content && length($ai_content) > 0) {
        $filtered++;
        next LINE;
    }

    # Transform AI content
    $ai_content = transform_ai($ai_content);

    # Transform Yent refs in user content too
    $user_content =~ s/Yent's/DOE's/g;
    $user_content =~ s/Yent/DOE/g;
    $user_content =~ s/yent/DOE/g;

    # Final safety check
    my $combined = $user_content . $ai_content;
    if ($combined =~ /Oleg/i || $combined =~ /Олег/ || $combined =~ /ataeff/i || $combined =~ /Иэнт/) {
        $filtered++;
        next LINE;
    }

    # Output in human/ai format
    print "{\"human\": \"$user_content\", \"ai\": \"$ai_content\"}\n";
    $kept++;
}

print STDERR "Total: $total | Kept: $kept | Filtered: $filtered\n";
