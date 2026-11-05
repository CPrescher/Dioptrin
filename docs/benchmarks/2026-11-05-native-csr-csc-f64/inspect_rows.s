	.build_version macos, 11, 0
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_masked
	.p2align	2
_masked:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	cmp	x1, #4
	b.hs	LBB0_5
	mov	x8, #0
	movi.2d	v0, #0000000000000000
	movi.2d	v1, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
LBB0_2:
	subs	x9, x1, x8
	b.eq	LBB0_28
	ldr	w10, [x0, x8, lsl #2]
	ldrb	w11, [x4, x10]
	cbz	w11, LBB0_15
	cmp	x9, #1
	b.ne	LBB0_16
	b	LBB0_28
LBB0_5:
	mov	x10, #0
	add	x11, x0, #8
	add	x9, x2, #16
	movi.2d	v0, #0000000000000000
	movi.2d	v1, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	b	LBB0_7
LBB0_6:
	add	x8, x10, #4
	add	x13, x10, #8
	add	x9, x9, #32
	mov	x11, x12
	mov	x10, x8
	cmp	x13, x1
	b.hi	LBB0_2
LBB0_7:
	ldur	w8, [x11, #-8]
	ldrb	w12, [x4, x8]
	cbz	w12, LBB0_11
	ldur	w8, [x11, #-4]
	ldrb	w12, [x4, x8]
	cbz	w12, LBB0_12
LBB0_9:
	mov	x12, x11
	ldr	w8, [x12], #16
	ldrb	w13, [x4, x8]
	cbz	w13, LBB0_13
LBB0_10:
	ldr	w8, [x11, #4]
	ldrb	w11, [x4, x8]
	cbnz	w11, LBB0_6
	b	LBB0_14
LBB0_11:
	ldur	d4, [x9, #-16]
	ldr	d5, [x6, x8, lsl #3]
	fmadd	d3, d4, d5, d3
	ldur	w8, [x11, #-4]
	ldrb	w12, [x4, x8]
	cbnz	w12, LBB0_9
LBB0_12:
	ldur	d4, [x9, #-8]
	ldr	d5, [x6, x8, lsl #3]
	fmadd	d2, d4, d5, d2
	mov	x12, x11
	ldr	w8, [x12], #16
	ldrb	w13, [x4, x8]
	cbnz	w13, LBB0_10
LBB0_13:
	ldr	d4, [x9]
	ldr	d5, [x6, x8, lsl #3]
	fmadd	d1, d4, d5, d1
	ldr	w8, [x11, #4]
	ldrb	w11, [x4, x8]
	cbnz	w11, LBB0_6
LBB0_14:
	ldr	d4, [x9, #8]
	ldr	d5, [x6, x8, lsl #3]
	fmadd	d0, d4, d5, d0
	b	LBB0_6
LBB0_15:
	ldr	d4, [x6, x10, lsl #3]
	ldr	d5, [x2, x8, lsl #3]
	fmadd	d3, d5, d4, d3
	cmp	x9, #1
	b.eq	LBB0_28
LBB0_16:
	orr	x10, x8, #0x1
	ldr	w11, [x0, x10, lsl #2]
	ldrb	w12, [x4, x11]
	cbz	w12, LBB0_18
	cmp	x9, #2
	b.ne	LBB0_19
	b	LBB0_28
LBB0_18:
	ldr	d4, [x6, x11, lsl #3]
	ldr	d5, [x2, x10, lsl #3]
	fmadd	d2, d5, d4, d2
	cmp	x9, #2
	b.eq	LBB0_28
LBB0_19:
	orr	x10, x8, #0x2
	ldr	w11, [x0, x10, lsl #2]
	ldrb	w12, [x4, x11]
	cbz	w12, LBB0_21
	cmp	x9, #3
	b.ne	LBB0_22
	b	LBB0_28
LBB0_21:
	ldr	d4, [x6, x11, lsl #3]
	ldr	d5, [x2, x10, lsl #3]
	fmadd	d1, d5, d4, d1
	cmp	x9, #3
	b.eq	LBB0_28
LBB0_22:
	orr	x10, x8, #0x3
	ldr	w11, [x0, x10, lsl #2]
	ldrb	w12, [x4, x11]
	cbnz	w12, LBB0_24
	ldr	d4, [x6, x11, lsl #3]
	ldr	d5, [x2, x10, lsl #3]
	fmadd	d0, d5, d4, d0
LBB0_24:
	cmp	x9, #4
	b.eq	LBB0_28
	sub	x9, x8, x1
	add	x10, x8, #4
	mov	x8, #-4
LBB0_26:
	ldr	w11, [x0, x10, lsl #2]
	ldrb	w11, [x4, x11]
	cbz	w11, LBB0_29
	sub	x8, x8, #1
	add	x10, x10, #1
	cmp	x9, x8
	b.ne	LBB0_26
LBB0_28:
	fadd	d2, d3, d2
	fadd	d0, d1, d0
	fadd	d0, d2, d0
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
LBB0_29:
	.cfi_restore_state
	neg	x0, x8
Lloh0:
	adrp	x2, l_anon.a9b051ddbcde552a64ab74a9f192d059.1@PAGE
Lloh1:
	add	x2, x2, l_anon.a9b051ddbcde552a64ab74a9f192d059.1@PAGEOFF
	mov	w1, #4
	bl	__RNvNtCs3bJEMBXHbwl_4core9panicking18panic_bounds_check
	.loh AdrpAdd	Lloh0, Lloh1
	.cfi_endproc

	.globl	_unmasked
	.p2align	2
_unmasked:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	cmp	x1, #4
	b.hs	LBB1_2
	mov	x8, #0
	movi.2d	v0, #0000000000000000
	movi.2d	v1, #0000000000000000
	b	LBB1_4
LBB1_2:
	mov	x8, #0
	add	x9, x0, #8
	movi.2d	v0, #0000000000000000
	mov	x10, x2
	movi.2d	v1, #0000000000000000
LBB1_3:
	ldp	w11, w12, [x9, #-8]
	add	x12, x4, x12, lsl #3
	ldp	w13, w14, [x9], #16
	ldp	q2, q3, [x10], #32
	ldr	d4, [x4, x11, lsl #3]
	ld1.d	{ v4 }[1], [x12]
	add	x11, x4, x14, lsl #3
	ldr	d5, [x4, x13, lsl #3]
	ld1.d	{ v5 }[1], [x11]
	fmla.2d	v1, v5, v3
	add	x11, x8, #8
	fmla.2d	v0, v4, v2
	add	x8, x8, #4
	cmp	x11, x1
	b.ls	LBB1_3
LBB1_4:
	mov	d2, v0[1]
	mov	d3, v1[1]
	subs	x9, x1, x8
	b.eq	LBB1_10
	ldr	w10, [x0, x8, lsl #2]
	ldr	d4, [x4, x10, lsl #3]
	ldr	d5, [x2, x8, lsl #3]
	fmadd	d0, d5, d4, d0
	cmp	x9, #1
	b.eq	LBB1_10
	orr	x10, x8, #0x1
	ldr	w11, [x0, x10, lsl #2]
	ldr	d4, [x4, x11, lsl #3]
	ldr	d5, [x2, x10, lsl #3]
	fmadd	d2, d5, d4, d2
	cmp	x9, #2
	b.eq	LBB1_10
	orr	x10, x8, #0x2
	ldr	w11, [x0, x10, lsl #2]
	ldr	d4, [x4, x11, lsl #3]
	ldr	d5, [x2, x10, lsl #3]
	fmadd	d1, d5, d4, d1
	cmp	x9, #3
	b.eq	LBB1_10
	cmp	x9, #4
	b.ne	LBB1_11
	orr	x8, x8, #0x3
	ldr	w9, [x0, x8, lsl #2]
	ldr	d4, [x4, x9, lsl #3]
	ldr	d5, [x2, x8, lsl #3]
	fmadd	d3, d5, d4, d3
LBB1_10:
	fadd	d0, d0, d2
	fadd	d1, d1, d3
	fadd	d0, d0, d1
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
LBB1_11:
	.cfi_restore_state
Lloh2:
	adrp	x2, l_anon.a9b051ddbcde552a64ab74a9f192d059.1@PAGE
Lloh3:
	add	x2, x2, l_anon.a9b051ddbcde552a64ab74a9f192d059.1@PAGEOFF
	mov	w0, #4
	mov	w1, #4
	bl	__RNvNtCs3bJEMBXHbwl_4core9panicking18panic_bounds_check
	.loh AdrpAdd	Lloh2, Lloh3
	.cfi_endproc

	.section	__TEXT,__cstring,cstring_literals
l_anon.a9b051ddbcde552a64ab74a9f192d059.0:
	.asciz	"/tmp/dioptrin-csr-csc-f64-20261105/inspect_rows.rs"

	.section	__DATA,__const
	.p2align	3, 0x0
l_anon.a9b051ddbcde552a64ab74a9f192d059.1:
	.quad	l_anon.a9b051ddbcde552a64ab74a9f192d059.0
	.asciz	"2\000\000\000\000\000\000\000_\000\000\000\021\000\000"

.subsections_via_symbols
