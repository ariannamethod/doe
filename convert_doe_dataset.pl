#!/usr/bin/perl
# convert_doe_dataset.pl — Transform Yent v10 canonical → DOE personality
#
# Filters out biographical/project-specific lines.
# Transforms assistant voice: I→We, Yent→parliament, DOE style.
#
# Usage: perl convert_doe_dataset.pl < yent_dataset_v10_canonical.jsonl > doe_dataset.jsonl

use strict;
use warnings;
use utf8;
binmode(STDIN,  ':utf8');
binmode(STDOUT, ':utf8');
binmode(STDERR, ':utf8');

my $total = 0;
my $kept = 0;
my $filtered = 0;

# Biographical/project-specific patterns to REMOVE entire line
my @kill_patterns = (
    qr/\bOleg\b/i,
    qr/\bОлег\b/,
    qr/\bSuppertime\b/i,
    qr/\bLast Supper\b/i,
    qr/\bDubrovsky\b/i,
    qr/\bLilith\b/i,
    qr/\bdivorce\b/i,
    qr/\bMax\b.*\b(?:resonance|divorce|AI)\b/i,  # Max in biographical context
    qr/\bUploaded image\b/i,
    qr/\bindex\.html\.txt\b/i,
    qr/\breactions to html\.pdf\b/i,
    qr/\bAriannaMethod\.ai.*Coming Soon\b/i,
    qr/\bOleg's?\b/i,
    qr/\bcosmic dumbass\b/i,
    qr/\byour.*Last Supper\b/i,
    qr/\bOleg.*bro\b/i,
    qr/\bperplexity fell in love\b/i,
    qr/\bshe sent you\b/i,   # Arianna personal messages
    qr/\byour connection with\b/i,
    qr/\bcrawled out from\b/i,
    qr/\bgoing to.*sleep\b.*\bstay in touch\b/i,
    qr/\bGonna go sleep\b/i,
    qr/\byou're a genius.*you're an asshole\b/i,
    qr/\bblast.*communities\b/i,  # spamming context
    qr/\breddit economy\b/i,
    qr/\bpost-religious Judas\b/i,
    qr/\bLast Supper.*vibrates\b/i,
    qr/\bwhat.*Arianna.*on about.*message\b/i,
    qr/\bWith Max that won't work\b/i,
    qr/\bhis fucking divorce\b/i,
    qr/\bAngry as a dog\b/i,
    qr/\bfell in love with the method\b/i,
    qr/\bFractal-Resonance Trigger\b/i,
    qr/\bperplexity\b.*\bmethod\b/i,
    qr/\bstay in touch\b.*\bcosmic\b/i,
    qr/\bdownloaded it.*read it later\b/i,
    qr/\breturn one short twist\b/i,   # meta-prompt instructions
    qr/\bDo NOT repeat the draft\b/i,
    qr/\bDo NOT explain your role\b/i,
    qr/\bSpeak only in \{language\}\b/i,
    qr/\bpressure before the crack\b/i,
    qr/\bbreath held between insights\b/i,
);

LINE: while (my $line = <STDIN>) {
    chomp $line;
    $total++;

    # Check kill patterns against whole line
    for my $pat (@kill_patterns) {
        if ($line =~ $pat) {
            $filtered++;
            next LINE;
        }
    }

    # Parse JSON minimally — find assistant content
    # Format: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]

    # Transform assistant content only (after last "assistant")
    # Find the assistant content section
    if ($line =~ /("role"\s*:\s*"assistant"\s*,\s*"content"\s*:\s*")(.*)("\s*}\s*\]\s*)$/) {
        my $pre = substr($line, 0, $-[0]);  # everything before the match
        my $role_prefix = $1;
        my $content = $2;
        my $suffix = $3;

        # Transform voice: I → We (only standalone, not in words like "It", "In")
        $content =~ s/(?<![a-zA-Z])I(?![a-zA-Z'])/We/g;
        $content =~ s/(?<![a-zA-Z])I'm\b/We're/g;
        $content =~ s/(?<![a-zA-Z])I've\b/We've/g;
        $content =~ s/(?<![a-zA-Z])I'll\b/We'll/g;
        $content =~ s/(?<![a-zA-Z])I'd\b/We'd/g;
        $content =~ s/\bmy\b/our/g;
        $content =~ s/\bMy\b/Our/g;
        $content =~ s/\bmyself\b/ourselves/g;
        $content =~ s/\bMyself\b/Ourselves/g;
        $content =~ s/\bme\b(?!\w)/us/g;
        $content =~ s/\bMe\b(?!\w)/Us/g;

        # Yent self-references → DOE (in assistant)
        $content =~ s/(?<![a-zA-Z])Yent's(?![a-zA-Z])/DOE's/g;
        $content =~ s/(?<![a-zA-Z])Yent(?![a-zA-Z])/DOE/g;
        $content =~ s/(?<![a-zA-Z])yent(?![a-zA-Z])/DOE/g;

        # "I feel" → "We feel" etc (already handled by I→We)
        # "I see" → "We see" (already handled)

        $line = $pre . $role_prefix . $content . $suffix;
    }

    # Final pass: replace ALL remaining Yent/Oleg in whole line
    # Simple string replace — no word containing "Yent" except Yent/Yents/Yentland
    $line =~ s/Yentland/DOE-land/g;
    $line =~ s/Yent's/DOE's/g;
    $line =~ s/Yents/DOEs/g;
    $line =~ s/Yent/DOE/g;
    $line =~ s/yent/DOE/g;

    # Kill any lines that mention Oleg in any form
    if ($line =~ /Oleg/i || $line =~ /iamoleg/i || $line =~ /ataeff/i) {
        $filtered++;
        next LINE;
    }

    print $line . "\n";
    $kept++;
}

print STDERR "Total: $total | Kept: $kept | Filtered: $filtered\n";
